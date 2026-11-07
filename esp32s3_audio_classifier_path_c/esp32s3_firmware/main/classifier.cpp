/**
 * @file classifier.cpp
 * @brief INT8 Audio Event Classifier implementation for ESP32-S3.
 *
 * Implements inference using either TensorFlow Lite for Microcontrollers (TFLM)
 * or fallback optimized INT8 spectral feature projection with EMA temporal smoothing.
 */

#include "model_data.h"
#include "classifier.h"
#include <math.h>
#include <string.h>
#include <algorithm>

extern const char* kCategoryLabels[4];
extern const float kFeatureNormMean[80];
extern const float kFeatureNormStd[80];
extern const float kInt8WeightScale;
extern const float kClassifierBias[4];
extern const int8_t kInt8StudentWeights[80 * 4];
extern const unsigned char g_student_model_tflite[];
extern const unsigned int g_student_model_tflite_len;

#if defined(USE_TFLITE_MICRO)
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#if USE_PSRAM_FOR_ARENA
#include "esp_heap_caps.h"
static uint8_t* s_tensor_arena = nullptr;
#else
alignas(16) static uint8_t s_tensor_arena[TENSOR_ARENA_SIZE_BYTES];
#endif
static tflite::MicroInterpreter* s_interpreter = nullptr;
static TfLiteTensor* s_input_tensor = nullptr;
static TfLiteTensor* s_output_tensor = nullptr;
#endif

AudioStudentClassifier::AudioStudentClassifier() : initialized_(false) {
    ema_probs_[0] = 0.0f;
    ema_probs_[1] = 0.0f;
    ema_probs_[2] = 0.0f;
    ema_probs_[3] = 1.0f;
}

bool AudioStudentClassifier::Init() {
#if defined(USE_TFLITE_MICRO)
    if (g_student_model_tflite_len > 64) {
#if USE_PSRAM_FOR_ARENA
        s_tensor_arena = (uint8_t*)heap_caps_malloc(TENSOR_ARENA_SIZE_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#endif
        const tflite::Model* model = tflite::GetModel(g_student_model_tflite);
        static tflite::MicroMutableOpResolver<8> resolver;
        resolver.AddConv2D();
        resolver.AddDepthwiseConv2D();
        resolver.AddFullyConnected();
        resolver.AddMean();
        resolver.AddSoftmax();
        resolver.AddReshape();
        resolver.AddQuantize();
        resolver.AddDequantize();

        static tflite::MicroInterpreter static_interp(model, resolver, s_tensor_arena, TENSOR_ARENA_SIZE_BYTES);
        s_interpreter = &static_interp;
        if (s_interpreter->AllocateTensors() == kTfLiteOk) {
            s_input_tensor = s_interpreter->input(0);
            s_output_tensor = s_interpreter->output(0);
        }
    }
#endif
    initialized_ = true;
    return true;
}

void AudioStudentClassifier::RunInt8SpectralStudent(const float* mel_patch_96x40, float* out_probs_4) {
    // Feature extraction: 40 temporal means and 40 temporal standard deviations across frames
    float feats[80];
    for (int m = 0; m < 40; ++m) {
        float sum = 0.0f;
        for (int t = 0; t < 96; ++t) {
            sum += mel_patch_96x40[t * 40 + m];
        }
        float mean = sum / 96.0f;
        float var_sum = 0.0f;
        for (int t = 0; t < 96; ++t) {
            float d = mel_patch_96x40[t * 40 + m] - mean;
            var_sum += d * d;
        }
        feats[m] = (mean - kFeatureNormMean[m]) / kFeatureNormStd[m];
        feats[40 + m] = (sqrtf(var_sum / 96.0f) - kFeatureNormMean[40 + m]) / kFeatureNormStd[40 + m];
    }

    // Quantized linear projection: INT8 weight matrix multiplication and scaled softmax
    float logits[4];
    float max_logit = -1e9f;
    for (int c = 0; c < 4; ++c) {
        float acc = 0.0f;
        for (int i = 0; i < 80; ++i) {
            acc += feats[i] * (float)kInt8StudentWeights[i * 4 + c];
        }
        logits[c] = acc * kInt8WeightScale + kClassifierBias[c];
        if (logits[c] > max_logit) max_logit = logits[c];
    }

    float exp_sum = 0.0f;
    for (int c = 0; c < 4; ++c) {
        out_probs_4[c] = expf((logits[c] - max_logit) * 2.5f);
        exp_sum += out_probs_4[c];
    }
    for (int c = 0; c < 4; ++c) {
        out_probs_4[c] /= exp_sum;
    }
}

ClassificationResult AudioStudentClassifier::ClassifyPatch(const float* mel_patch_96x40, float rms_energy) {
    ClassificationResult res;
    res.rms_energy = rms_energy;

    // 1. Adaptive RMS energy silence gating
    if (rms_energy < RMS_SILENCE_THRESHOLD) {
        res.raw_probs[0] = 0.01f;
        res.raw_probs[1] = 0.01f;
        res.raw_probs[2] = 0.01f;
        res.raw_probs[3] = 0.97f;
    } else {
        bool ran_tflm = false;
#if defined(USE_TFLITE_MICRO)
        if (s_interpreter != nullptr && s_input_tensor != nullptr && g_student_model_tflite_len > 64) {
            const float in_scale = s_input_tensor->params.scale;
            const int32_t in_zp = s_input_tensor->params.zero_point;
            for (int i = 0; i < 96 * 40; ++i) {
                int32_t q = (int32_t)roundf(mel_patch_96x40[i] / in_scale) + in_zp;
                if (q < -128) q = -128;
                if (q > 127) q = 127;
                s_input_tensor->data.int8[i] = (int8_t)q;
            }
            if (s_interpreter->Invoke() == kTfLiteOk) {
                const float out_scale = s_output_tensor->params.scale;
                const int32_t out_zp = s_output_tensor->params.zero_point;
                for (int c = 0; c < 4; ++c) {
                    res.raw_probs[c] = (s_output_tensor->data.int8[c] - out_zp) * out_scale;
                }
                ran_tflm = true;
            }
        }
#endif
        if (!ran_tflm) {
            RunInt8SpectralStudent(mel_patch_96x40, res.raw_probs);
        }
    }

    // 2. Exponential Moving Average (EMA) temporal smoothing
    for (int c = 0; c < 4; ++c) {
        ema_probs_[c] = EMA_SMOOTHING_ALPHA * res.raw_probs[c] + (1.0f - EMA_SMOOTHING_ALPHA) * ema_probs_[c];
        res.smoothed_probs[c] = ema_probs_[c];
    }

    // 3. Class selection and hysteresis confidence thresholding
    int best_c = 3;
    float best_p = res.smoothed_probs[3];
    for (int c = 0; c < 3; ++c) {
        if (res.smoothed_probs[c] > best_p) {
            best_p = res.smoothed_probs[c];
            best_c = c;
        }
    }

    const float thresholds[3] = {THRESHOLD_DOG, THRESHOLD_DJ, THRESHOLD_TRAFFIC};
    if (best_c < 3 && best_p >= thresholds[best_c]) {
        res.predicted_class = best_c;
        res.label = kCategoryLabels[best_c];
        res.confidence = best_p;
        res.triggered = true;
    } else {
        res.predicted_class = 3;
        res.label = kCategoryLabels[3];
        res.confidence = res.smoothed_probs[3];
        res.triggered = false;
    }

    return res;
}
