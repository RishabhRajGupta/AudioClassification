#pragma once

/**
 * @file config.h
 * @brief System hardware pin allocation, DSP hyperparameters, and memory configuration.
 *
 * Configured for ESP32-S3 microcontroller with INMP441 I2S microphone and GPIO indicators.
 */

// ============================================================================
// 1. I2S MEMS MICROPHONE PINOUT (INMP441 / ICS-43434 / SPH0645)
// ============================================================================
#define I2S_MIC_BCLK_GPIO       4     // Serial Clock (SCK / BCLK)
#define I2S_MIC_WS_GPIO         5     // Word Select (WS / LRCLK)
#define I2S_MIC_DIN_GPIO        6     // Serial Data In (SD / DOUT from mic)
#define I2S_MIC_SAMPLE_RATE     16000 // 16 kHz mono

// ============================================================================
// 2. HARDWARE OUTPUT GPIO ALLOCATION (Acoustic Event Status Indicators)
// ============================================================================
#define LED_DOG_GPIO            15    // GPIO pin assignment for 'Dog' acoustic event
#define LED_TRAFFIC_GPIO        16    // GPIO pin assignment for 'Traffic' acoustic event
#define LED_DJ_GPIO             17    // GPIO pin assignment for 'DJ' acoustic event
#define LED_HOLD_DURATION_MS    1000  // Pulse assertion duration (ms) for output indicators

// ============================================================================
// 3. AUDIO DSP & INFERENCE HYPERPARAMETERS
// ============================================================================
#define AUDIO_PATCH_SAMPLES     15360 // 0.96s window (96 frames * 160 hop)
#define AUDIO_HOP_SAMPLES       3840  // 0.24s stride (24 new frames -> ~4.1 Hz update rate)
#define RMS_SILENCE_THRESHOLD   0.008f// Adaptive RMS gate to ignore room silence
#define EMA_SMOOTHING_ALPHA     0.60f // 3-window Exponential Moving Average weight

// Empirical classification confidence thresholds for hysteresis decision logic
#define THRESHOLD_DOG           0.52f
#define THRESHOLD_DJ            0.52f
#define THRESHOLD_TRAFFIC       0.45f

// ============================================================================
// 4. MEMORY CONFIGURATION (512 KB Internal SRAM vs. 8 MB PSRAM)
// ============================================================================
// Tensor arena allocation (96 KB). Sized for internal SRAM execution.
// Set USE_PSRAM_FOR_ARENA to 1 when evaluating extended model topologies exceeding internal heap.
#define TENSOR_ARENA_SIZE_BYTES (96 * 1024)
#define USE_PSRAM_FOR_ARENA     0
