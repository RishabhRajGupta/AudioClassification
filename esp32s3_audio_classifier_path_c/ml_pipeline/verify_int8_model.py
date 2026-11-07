#!/usr/bin/env python3
"""
Verifies the quantized `model_int8.tflite` on 16 kHz WAV files using `tf.lite.Interpreter`
(simulating the exact int8 input quantization and output dequantization performed on the ESP32-S3).
"""
import os
import sys
import numpy as np
from dsp_mel_extractor import load_wav_16k_mono, extract_log_mel_patch

CLASS_NAMES = ["Dog", "DJ", "Traffic", "Background"]

def evaluate_tflite_int8(tflite_path: str, wav_path: str):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    wav = load_wav_16k_mono(wav_path)
    mel_patch = extract_log_mel_patch(wav)[np.newaxis, :, :, np.newaxis]  # (1, 96, 40, 1)

    q_in = np.clip(np.round(mel_patch / in_scale + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(inp["index"], q_in)
    interp.invoke()
    q_out = interp.get_tensor(out["index"])[0]
    probs = (q_out.astype(np.float32) - out_zp) * out_scale

    pred_idx = int(np.argmax(probs))
    print(f"File: {os.path.basename(wav_path):24s} -> Predicted: {CLASS_NAMES[pred_idx]:10s} ({probs[pred_idx]*100:5.1f}%) | Probs: {np.round(probs, 3)}")

if __name__ == "__main__":
    tflite_file = os.path.join(os.path.dirname(__file__), "model_int8.tflite")
    if not os.path.isfile(tflite_file):
        print("Error: Target model 'model_int8.tflite' not found. Execute distill_from_yamnet.py to produce the quantized model.")
        sys.exit(1)
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_audio")
    for fn in sorted(os.listdir(sample_dir)):
        if fn.endswith(".wav"):
            evaluate_tflite_int8(tflite_file, os.path.join(sample_dir, fn))
