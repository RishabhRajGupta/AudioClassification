/**
 * @file esp32s3_audio_classifier.ino
 * @brief Arduino framework entry point for ESP32-S3 edge acoustic event classifier.
 *
 * Hardware Target Configuration:
 *   - Board: ESP32S3 Dev Module
 *   - Flash Size: 8 MB
 *   - PSRAM: Octal PSRAM (Optional; student model executes within internal SRAM)
 *   - CPU Frequency: 240 MHz
 */
#include <Arduino.h>
#include <driver/i2s.h>
#include "../main/config.h"
#include "../main/mel_spectrogram.h"
#include "../main/classifier.h"
#include "../main/mel_spectrogram.cpp"
#include "../main/classifier.cpp"
#include "../main/model_data.cc"

static float s_audio_ring[AUDIO_PATCH_SAMPLES + MEL_WIN_LEN];
static float s_mel_patch[MEL_NUM_FRAMES * MEL_NUM_BANDS];
static int16_t s_hop_pcm16[AUDIO_HOP_SAMPLES];
static MelSpectrogramExtractor s_mel_extractor;
static AudioStudentClassifier s_classifier;
static unsigned long s_led_off_ms[3] = {0, 0, 0}; // [0: Dog, 1: DJ, 2: Traffic]
static const int kLedPins[3] = {LED_DOG_GPIO, LED_DJ_GPIO, LED_TRAFFIC_GPIO};

void setup() {
    Serial.begin(115200);
    for (int i = 0; i < 3; ++i) {
        pinMode(kLedPins[i], OUTPUT);
        digitalWrite(kLedPins[i], LOW);
    }

    i2s_config_t i2s_cfg = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = I2S_MIC_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false
    };
    i2s_pin_config_t pin_cfg = {
        .bck_io_num = I2S_MIC_BCLK_GPIO,
        .ws_io_num = I2S_MIC_WS_GPIO,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_DIN_GPIO
    };
    i2s_driver_install(I2S_NUM_0, &i2s_cfg, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_cfg);
    s_classifier.Init();
}

void loop() {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, s_hop_pcm16, sizeof(s_hop_pcm16), &bytes_read, portMAX_DELAY);
    int samples_read = bytes_read / sizeof(int16_t);
    if (samples_read <= 0) return;

    const int total_buf = AUDIO_PATCH_SAMPLES + MEL_WIN_LEN;
    memmove(s_audio_ring, s_audio_ring + samples_read, (total_buf - samples_read) * sizeof(float));

    float sq_sum = 0.0f;
    for (int i = 0; i < samples_read; ++i) {
        float s = (float)s_hop_pcm16[i] / 32768.0f;
        s_audio_ring[total_buf - samples_read + i] = s;
        sq_sum += s * s;
    }
    float rms = sqrtf(sq_sum / (float)samples_read);

    unsigned long t0 = millis();
    s_mel_extractor.ComputePatch96x40(s_audio_ring, s_mel_patch);
    ClassificationResult res = s_classifier.ClassifyPatch(s_mel_patch, rms);
    unsigned long latency_ms = millis() - t0;

    unsigned long now = millis();
    if (res.triggered && res.predicted_class < 3) {
        s_led_off_ms[res.predicted_class] = now + LED_HOLD_DURATION_MS;
    }
    for (int i = 0; i < 3; ++i) {
        digitalWrite(kLedPins[i], (now < s_led_off_ms[i]) ? HIGH : LOW);
    }

    Serial.printf("{\"label\":\"%s\",\"triggered\":%s,\"confidence\":%.3f,\"probs\":[%.3f,%.3f,%.3f,%.3f],\"rms\":%.4f,\"latency_ms\":%lu,\"leds\":[%d,%d,%d]}\n",
                  res.label, res.triggered ? "true" : "false", res.confidence,
                  res.smoothed_probs[0], res.smoothed_probs[1], res.smoothed_probs[2], res.smoothed_probs[3],
                  res.rms_energy, latency_ms,
                  digitalRead(LED_DOG_GPIO), digitalRead(LED_TRAFFIC_GPIO), digitalRead(LED_DJ_GPIO));
}
