#pragma once
#include <stdint.h>
#include "config.h"

struct ClassificationResult {
    int predicted_class;        // 0: Dog, 1: DJ, 2: Traffic, 3: Background
    const char* label;          // "Dog", "DJ", "Traffic", or "Background"
    float confidence;           // Smoothed probability of winning class
    float raw_probs[4];         // Instantaneous [Dog, DJ, Traffic, Background]
    float smoothed_probs[4];    // EMA-smoothed [Dog, DJ, Traffic, Background]
    float rms_energy;           // Audio window RMS amplitude
    bool triggered;             // True if winning class exceeds threshold
};

class AudioStudentClassifier {
public:
    AudioStudentClassifier();
    bool Init();
    ClassificationResult ClassifyPatch(const float* mel_patch_96x40, float rms_energy);

private:
    void RunInt8SpectralStudent(const float* mel_patch_96x40, float* out_probs_4);
    float ema_probs_[4];
    bool initialized_;
};
