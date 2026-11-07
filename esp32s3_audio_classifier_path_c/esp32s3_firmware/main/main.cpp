/**
 * @file main.cpp
 * @brief ESP32-S3 Real-Time Edge Audio Classification System entry point.
 *
 * Ingests audio via I2S DMA, extracts Log-Mel features, executes INT8 inference,
 * asserts GPIO status indicators, and transmits UART telemetry payloads.
 */

#include <stdio.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "mel_spectrogram.h"
#include "classifier.h"

#ifdef ESP_PLATFORM
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

static const char* TAG = "ESP32S3_AUDIO_AI";

// Rolling 15,600-sample audio ring buffer in internal SRAM (~62 KB)
static float s_audio_ring[AUDIO_PATCH_SAMPLES + MEL_WIN_LEN];
static float s_mel_patch[MEL_NUM_FRAMES * MEL_NUM_BANDS];
static MelSpectrogramExtractor s_mel_extractor;
static AudioStudentClassifier s_classifier;
static i2s_chan_handle_t s_rx_handle = NULL;

// State tracking for non-blocking GPIO indicator assertion hold intervals
static int64_t s_led_off_deadline_us[3] = {0, 0, 0};
static const gpio_num_t kLedGpios[3] = {
    (gpio_num_t)LED_DOG_GPIO,     // Index 0: Dog
    (gpio_num_t)LED_DJ_GPIO,      // Index 1: DJ
    (gpio_num_t)LED_TRAFFIC_GPIO  // Index 2: Traffic
};

static void InitPhysicalLeds() {
    for (int i = 0; i < 3; ++i) {
        gpio_reset_pin(kLedGpios[i]);
        gpio_set_direction(kLedGpios[i], GPIO_MODE_OUTPUT);
        gpio_set_level(kLedGpios[i], 0);
    }
}

static void UpdatePhysicalLeds(int triggered_class) {
    int64_t now_us = esp_timer_get_time();
    if (triggered_class >= 0 && triggered_class < 3) {
        s_led_off_deadline_us[triggered_class] = now_us + (int64_t)LED_HOLD_DURATION_MS * 1000LL;
    }
    for (int i = 0; i < 3; ++i) {
        gpio_set_level(kLedGpios[i], (now_us < s_led_off_deadline_us[i]) ? 1 : 0);
    }
}

static void InitI2sMicrophone() {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = 512;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &s_rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(I2S_MIC_SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = (gpio_num_t)I2S_MIC_BCLK_GPIO,
            .ws   = (gpio_num_t)I2S_MIC_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = (gpio_num_t)I2S_MIC_DIN_GPIO,
            .invert_flags = { .mclk_inv = false, .bclk_inv = false, .ws_inv = false },
        },
    };
    ESP_ERROR_CHECK(i2s_channel_init_std_mode(s_rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(s_rx_handle));
}

static void AudioInferenceTask(void* arg) {
    static int16_t hop_pcm16[AUDIO_HOP_SAMPLES];
    size_t bytes_read = 0;

    while (true) {
        // Read 0.24s hop (3,840 samples) from I2S DMA
        esp_err_t err = i2s_channel_read(s_rx_handle, hop_pcm16, sizeof(hop_pcm16), &bytes_read, portMAX_DELAY);
        if (err != ESP_OK || bytes_read == 0) continue;

        int samples_read = bytes_read / sizeof(int16_t);
        const int total_buf = AUDIO_PATCH_SAMPLES + MEL_WIN_LEN;
        memmove(s_audio_ring, s_audio_ring + samples_read, (total_buf - samples_read) * sizeof(float));

        float sq_sum = 0.0f;
        for (int i = 0; i < samples_read; ++i) {
            float sample = (float)hop_pcm16[i] / 32768.0f;
            s_audio_ring[total_buf - samples_read + i] = sample;
            sq_sum += sample * sample;
        }
        float rms = sqrtf(sq_sum / (float)samples_read);

        int64_t t0 = esp_timer_get_time();
        s_mel_extractor.ComputePatch96x40(s_audio_ring, s_mel_patch);
        ClassificationResult res = s_classifier.ClassifyPatch(s_mel_patch, rms);
        int64_t latency_ms = (esp_timer_get_time() - t0) / 1000;

        // 1. Assert physical indicator GPIO lines corresponding to classified acoustic event
        UpdatePhysicalLeds(res.triggered ? res.predicted_class : -1);

        // 2. Transmit serialized JSON telemetry payload over UART interface (115200 baud)
        printf("{\"label\":\"%s\",\"triggered\":%s,\"confidence\":%.3f,\"probs\":[%.3f,%.3f,%.3f,%.3f],\"rms\":%.4f,\"latency_ms\":%lld,\"leds\":[%d,%d,%d]}\n",
               res.label,
               res.triggered ? "true" : "false",
               res.confidence,
               res.smoothed_probs[0], res.smoothed_probs[1], res.smoothed_probs[2], res.smoothed_probs[3],
               res.rms_energy,
               latency_ms,
               gpio_get_level(kLedGpios[0]),
               gpio_get_level(kLedGpios[2]), // Traffic
               gpio_get_level(kLedGpios[1])  // DJ
        );
    }
}

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "Initializing ESP32-S3 Audio Event Classifier (Dog | DJ | Traffic)...");
    InitPhysicalLeds();
    InitI2sMicrophone();
    s_classifier.Init();
    xTaskCreatePinnedToCore(AudioInferenceTask, "AudioInferenceTask", 16384, NULL, 5, NULL, 1);
}
#endif
