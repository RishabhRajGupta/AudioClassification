#!/usr/bin/env python3
"""
Exact 16 kHz 96x40 Log-Mel Spectrogram (MFE) Extractor for ESP32-S3.
Matches `esp32s3_firmware/main/mel_spectrogram.cpp` bit-for-bit:
- Sample rate: 16,000 Hz mono
- Window length: 400 samples (25 ms Hann window)
- Frame hop: 160 samples (10 ms stride) -> 96 frames = 0.96s patch (15,360 samples)
- FFT size: 512 points (257 positive frequency bins)
- Mel filterbank: 40 HTK triangular bands from 125 Hz to 7,500 Hz
- Log scaling: log(max(mel_energy, 1e-6))
"""
import wave
import numpy as np
from scipy import signal

SR = 16000
PATCH_SAMPLES = 15360
WIN_LEN = 400
HOP_LEN = 160
N_FFT = 512
N_MELS = 40
FMIN = 125.0
FMAX = 7500.0

def hz_to_mel_htk(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz_htk(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def build_mel_filterbank() -> np.ndarray:
    mel_min = hz_to_mel_htk(FMIN)
    mel_max = hz_to_mel_htk(FMAX)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = mel_to_hz_htk(mel_points)
    fft_freqs = np.linspace(0, SR / 2.0, N_FFT // 2 + 1)
    weights = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(N_MELS):
        f_left, f_center, f_right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k, f in enumerate(fft_freqs):
            if f_left <= f <= f_center and f_center > f_left:
                weights[m, k] = (f - f_left) / (f_center - f_left)
            elif f_center < f <= f_right and f_right > f_center:
                weights[m, k] = (f_right - f) / (f_right - f_center)
    return weights

HANN_WINDOW = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN_LEN) / WIN_LEN)).astype(np.float32)
MEL_BASIS = build_mel_filterbank()

def load_wav_16k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    if framerate != SR:
        num_target = int(len(data) * SR / framerate)
        data = signal.resample(data, num_target).astype(np.float32)
    return data

def extract_log_mel_patch(waveform_f32: np.ndarray) -> np.ndarray:
    """Returns a float32 array of shape (96, 40) representing 0.96s of audio."""
    req_len = PATCH_SAMPLES + WIN_LEN
    if len(waveform_f32) < req_len:
        waveform_f32 = np.pad(waveform_f32, (0, req_len - len(waveform_f32)))
    patch = np.zeros((96, N_MELS), dtype=np.float32)
    for t in range(96):
        start = t * HOP_LEN
        frame = waveform_f32[start : start + WIN_LEN] * HANN_WINDOW
        padded = np.zeros(N_FFT, dtype=np.float32)
        padded[:WIN_LEN] = frame
        spec = np.fft.rfft(padded)
        power = (spec.real ** 2 + spec.imag ** 2) / N_FFT
        mel_energy = np.dot(MEL_BASIS, power)
        patch[t, :] = np.log(np.maximum(mel_energy, 1e-6))
    return patch

if __name__ == "__main__":
    import os
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_audio")
    for fname in sorted(os.listdir(sample_dir)):
        if fname.endswith(".wav"):
            wav = load_wav_16k_mono(os.path.join(sample_dir, fname))
            patch = extract_log_mel_patch(wav)
            print(f"{fname:24s} -> Log-Mel shape: {patch.shape}, min={patch.min():.2f}, max={patch.max():.2f}")
