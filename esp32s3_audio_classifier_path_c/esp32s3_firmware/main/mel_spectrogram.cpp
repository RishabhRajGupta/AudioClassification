#include "mel_spectrogram.h"
#include <math.h>
#include <string.h>

MelSpectrogramExtractor::MelSpectrogramExtractor() {
    memset(fft_real_, 0, sizeof(fft_real_));
    memset(fft_imag_, 0, sizeof(fft_imag_));
    memset(power_spec_, 0, sizeof(power_spec_));
}

void MelSpectrogramExtractor::Radix2FFT512(float* real, float* imag) {
    // Bit-reversal permutation for N = 512
    const int N = MEL_FFT_SIZE;
    int j = 0;
    for (int i = 0; i < N; ++i) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int m = N >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    // Cooley-Tukey butterflies
    for (int s = 1; s <= 9; ++s) {
        int m = 1 << s;
        int half_m = m >> 1;
        float theta = -2.0f * 3.14159265358979323846f / (float)m;
        float w_m_r = cosf(theta);
        float w_m_i = sinf(theta);
        for (int k = 0; k < N; k += m) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            for (int m_idx = 0; m_idx < half_m; ++m_idx) {
                int t_idx = k + m_idx + half_m;
                int u_idx = k + m_idx;
                float t_r = w_r * real[t_idx] - w_i * imag[t_idx];
                float t_i = w_r * imag[t_idx] + w_i * real[t_idx];
                real[t_idx] = real[u_idx] - t_r;
                imag[t_idx] = imag[u_idx] - t_i;
                real[u_idx] += t_r;
                imag[u_idx] += t_i;
                float next_w_r = w_r * w_m_r - w_i * w_m_i;
                w_i = w_r * w_m_i + w_i * w_m_r;
                w_r = next_w_r;
            }
        }
    }
}

void MelSpectrogramExtractor::ComputeSingleFrame(const float* frame_400, float* out_mel_40) {
    for (int i = 0; i < MEL_WIN_LEN; ++i) {
        fft_real_[i] = frame_400[i] * kHannWindow[i];
        fft_imag_[i] = 0.0f;
    }
    for (int i = MEL_WIN_LEN; i < MEL_FFT_SIZE; ++i) {
        fft_real_[i] = 0.0f;
        fft_imag_[i] = 0.0f;
    }

    Radix2FFT512(fft_real_, fft_imag_);

    const float inv_n = 1.0f / (float)MEL_FFT_SIZE;
    for (int k = 0; k <= MEL_FFT_SIZE / 2; ++k) {
        power_spec_[k] = (fft_real_[k] * fft_real_[k] + fft_imag_[k] * fft_imag_[k]) * inv_n;
    }

    // Multiply by sparse 40-band HTK Mel filterbank & apply log()
    for (int m = 0; m < MEL_NUM_BANDS; ++m) {
        uint16_t start_bin = kMelBandStartBin[m];
        uint16_t num_bins = kMelBandNumBins[m];
        uint16_t offset = kMelBandWeightOffset[m];
        float energy = 0.0f;
        for (uint16_t b = 0; b < num_bins; ++b) {
            energy += power_spec_[start_bin + b] * kMelSparseWeights[offset + b];
        }
        if (energy < 1e-6f) energy = 1e-6f;
        out_mel_40[m] = logf(energy);
    }
}

void MelSpectrogramExtractor::ComputePatch96x40(const float* waveform_15600, float* out_patch_3840) {
    for (int t = 0; t < MEL_NUM_FRAMES; ++t) {
        const float* frame_ptr = waveform_15600 + (t * MEL_HOP_LEN);
        float* out_row = out_patch_3840 + (t * MEL_NUM_BANDS);
        ComputeSingleFrame(frame_ptr, out_row);
    }
}
