#pragma once
#include <stdint.h>
#include "mel_filterbank_lut.h"

class MelSpectrogramExtractor {
public:
    MelSpectrogramExtractor();
    // Computes a single 40-band Log-Mel frame from 400 float32 audio samples [-1.0, 1.0]
    void ComputeSingleFrame(const float* frame_400, float* out_mel_40);
    // Computes a full [96 x 40] Log-Mel patch from 15,600 float32 samples
    void ComputePatch96x40(const float* waveform_15600, float* out_patch_3840);

private:
    void Radix2FFT512(float* real, float* imag);
    float fft_real_[MEL_FFT_SIZE];
    float fft_imag_[MEL_FFT_SIZE];
    float power_spec_[MEL_FFT_SIZE / 2 + 1];
};
