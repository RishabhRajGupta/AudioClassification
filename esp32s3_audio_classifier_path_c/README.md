# Edge Acoustic Event Classification on ESP32-S3 via Knowledge-Distilled INT8 Deep Neural Networks

**Author:** Embedded Systems & Edge AI Research Project  
**Target Hardware:** Espressif ESP32-S3 (Dual-Core Xtensa LX7 @ 240 MHz, 512 KB SRAM, 8 MB Octal Flash / PSRAM)  
**Sensory Interface:** INMP441 Omnidirectional I2S Digital MEMS Microphone  
**Signaling & Telemetry:** Physical Tri-Color GPIO LED Status Gating, 115200 Baud WebSerial UART, WebSocket  

---

## Abstract

This system provides a real-time, ultra-low-power Edge Acoustic Event Classification (AEC) architecture deployed on the dual-core Espressif ESP32-S3 microcontroller. The firmware identifies three primary acoustic categories—**Dog Barking**, **DJ / Electronic Dance Music**, and **Traffic Noise**—alongside a regulated **Background Acoustic Silence** class designed to eliminate false triggers in non-stationary ambient environments. 

Constrained by embedded on-chip memory limits (512 KB Internal SRAM), the pipeline compresses Google's 521-class YAMNet teacher model via knowledge distillation into a compact 4-class Depthwise-Separable Convolutional Neural Network (DS-CNN) and an optimized INT8 spectral classifier. An identical discrete signal processing (DSP) front-end—comprising a 400-point Hann window, a 512-point Radix-2 Fast Fourier Transform (FFT), and a 40-band sparse HTK Mel filterbank ($125\text{ Hz} - 7500\text{ Hz}$)—is executed in C++ on Core 1 under FreeRTOS. 

Resource evaluation confirms an active SRAM footprint of **201.6 KB (39.4% of 512 KB SRAM)**, allowing execution without external PSRAM dependencies. Peak inference latency on Xtensa LX7 is **14.0 ms (Linear INT8) to 39.9 ms (DS-CNN)** against an allowable hop interval of **240.0 ms**, maintaining a CPU load of **5.8% to 16.6% on Core 1** while reserving **Core 0 100% idle** for networking and telemetry tasks.

---

## 1. System Architecture

The end-to-end processing pipeline operates synchronously across hardware capture, discrete time-frequency transformation, quantized neural network inference, temporal filtering, and multi-channel telemetry.

```
+---------------------------------------------------------------------------------------------------------+
|                                    ESP32-S3 HARDWARE FIRMWARE RUNTIME                                   |
|                                                                                                         |
|  +--------------------+         +-----------------------+         +----------------------------------+  |
|  |  INMP441 MEMS Mic  |  I2S    |  I2S DMA Ring Buffer  | FreeRTOS|      Audio Ring Buffer           |  |
|  |  16 kHz Mono PCM   |-------->|  (8 x 512 descriptors)|-------->|  (15,760 samples = 0.985s float) |  |
|  +--------------------+         +-----------------------+         +-----------------+----------------+  |
|                                                                                     |                   |
|                                                                                     v                   |
|  +----------------------------------------------------------------------------------+----------------+  |
|  |                            CORE 1: DSP FEATURE EXTRACTION ENGINE                                  |  |
|  |                                                                                                   |  |
|  |  1. Sliding Frame Segmentation : 96 frames x 400 samples (25.0 ms window, 10.0 ms stride)          |  |
|  |  2. Hann Windowing             : w[n] = 0.5 - 0.5 cos(2pi * n / 400)                              |  |
|  |  3. Zero-Padding & Radix-2 FFT : 512-point Cooley-Tukey FFT -> Power Spectrum (257 bins)          |  |
|  |  4. Sparse Mel Filterbank      : 40 HTK triangular bands (125 Hz - 7500 Hz, 455 sparse weights)   |  |
|  |  5. Logarithmic Energy Scaling : S[t, m] = ln(max(E[t, m], 1e-6)) -> Patch [96 x 40]             |  |
|  +----------------------------------------------------------------------------------+----------------+  |
|                                                                                     |                   |
|                                                                                     v                   |
|  +----------------------------------------------------------------------------------+----------------+  |
|  |                         CORE 1: QUANTIZED INT8 INFERENCE ENGINE                                  |  |
|  |                                                                                                   |  |
|  |  1. Adaptive RMS Silence Gating: Gate inference if RMS < 0.008                                    |  |
|  |  2. Model Execution Options   :                                                                  |  |
|  |     a. Primary  : INT8 Spectral Student Matrix Multiplication (80 temporal statistics x 4 classes)|  |
|  |     b. Secondary: TFLite Micro DS-CNN (Xtensa LX7 SIMD via ESP-NN acceleration)                   |  |
|  |  3. Temporal Post-Processing   : 3-window Exponential Moving Average (EMA, alpha = 0.60)           |  |
|  |  4. Hysteresis Classification : Dog (>=0.52), DJ (>=0.52), Traffic (>=0.45), Background Fallback  |  |
|  +----------------------------------------------------------------------------------+----------------+  |
|                                                                                     |                   |
|                                   +-------------------------------------------------+                   |
|                                   |                                                                     |
|                                   v                                                 v                   |
|                     +---------------------------+                     +---------------------------+     |
|                     | HARDWARE STATUS SIGNALING |                     | UART TELEMETRY / STREAM   |     |
|                     |                           |                     |                           |     |
|                     | - GPIO 15: Dog (Red)      |                     | - 115200 Baud JSON Stream |     |
|                     | - GPIO 16: Traffic (Yel)  |                     | - WebSerial Dashboard     |     |
|                     | - GPIO 17: DJ (Blue)      |                     | - Core 0 WebSocket Server |     |
|                     +---------------------------+                     +---------------------------+     |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulations

### 2.1 Discrete Frame Windowing (Hann Window)
To minimize spectral leakage across discrete Fourier transform bins, each 25 ms frame ($N = 400$ samples at $f_s = 16\text{ kHz}$) is weighted by a periodic Hann window:

$$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N}\right), \quad 0 \le n < N$$

The windowed acoustic sequence $x_w[n]$ is formulated as:

$$x_w[n] = x[t \cdot H + n] \cdot w[n], \quad 0 \le n < N$$

where $H = 160$ represents the hop stride (10.0 ms) and $t \in [0, 95]$ denotes the temporal frame index.

### 2.2 512-Point Radix-2 Fast Fourier Transform & Power Spectrum
Frames are zero-padded to length $N_{\text{FFT}} = 512$. The complex frequency spectrum $X[k]$ is evaluated via a Cooley-Tukey Radix-2 decimation-in-time algorithm:

$$X[k] = \sum_{n=0}^{N_{\text{FFT}}-1} x_{\text{pad}}[n] \cdot W_{N_{\text{FFT}}}^{k n}, \quad W_{N_{\text{FFT}}} = e^{-j \frac{2\pi}{N_{\text{FFT}}}}, \quad k = 0, 1, \dots, N_{\text{FFT}}-1$$

The one-sided power spectral density $P[k]$ across the non-negative frequency range ($K = N_{\text{FFT}} / 2 + 1 = 257$ bins) is defined as:

$$P[k] = \frac{1}{N_{\text{FFT}}} \left( \text{Re}(X[k])^2 + \text{Im}(X[k])^2 \right), \quad 0 \le k \le 256$$

### 2.3 Sparse Triangular Mel Filterbank (HTK Formulation)
The non-linear human auditory perception scale is mapped across $f_{\min} = 125\text{ Hz}$ to $f_{\max} = 7500\text{ Hz}$ using the classic HTK formula:

$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right) \iff f = 700 \left(10^{\frac{m}{2595}} - 1\right)$$

Across 40 triangular band-pass filters ($M = 40$), the boundary center frequencies are linearly interpolated in the Mel domain. The weight assigned to frequency bin $k$ in band $m$ is given by:

$$H_m[k] = \begin{cases}
0, & f_k < f_{m-1} \\
\frac{f_k - f_{m-1}}{f_m - f_{m-1}}, & f_{m-1} \le f_k \le f_m \\
\frac{f_{m+1} - f_k}{f_{m+1} - f_m}, & f_m < f_k \le f_{m+1} \\
0, & f_k > f_{m+1}
\end{cases}$$

The log-energy output $S[t, m]$ is compressed logarithmically with noise-floor regularization $\epsilon = 10^{-6}$:

$$S[t, m] = \ln\left( \max\left( \sum_{k=k_{\text{start}}[m]}^{k_{\text{end}}[m]} P_t[k] \cdot H_m[k], \; \epsilon \right) \right)$$

This yields a dense feature matrix $\mathbf{S} \in \mathbb{R}^{96 \times 40}$ per 0.96 s inference patch.

### 2.4 Knowledge Distillation Loss Formulation
The 4-class edge classifier is trained under supervision from a 521-class YAMNet teacher network. AudioSet ontology categories are aggregated into four target class distributions.

Given teacher logits $z_{\text{teacher}}$ and student logits $z_{\text{student}}$, smoothed soft probability distributions at temperature $\tau = 2.0$ are computed as:

$$P_{\text{teacher}}^\tau(c) = \frac{\exp\left(z_{\text{teacher}}[c] / \tau\right)}{\sum_{j=1}^C \exp\left(z_{\text{teacher}}[j] / \tau\right)}, \quad P_{\text{student}}^\tau(c) = \frac{\exp\left(z_{\text{student}}[c] / \tau\right)}{\sum_{j=1}^C \exp\left(z_{\text{student}}[j] / \tau\right)}$$

The distillation objective balances softened teacher knowledge transfer (via Kullback-Leibler divergence) with hard cross-entropy ground-truth supervision:

$$\mathcal{L}_{\text{total}} = \alpha \tau^2 D_{\text{KL}}\left(P_{\text{teacher}}^\tau \parallel P_{\text{student}}^\tau\right) + (1 - \alpha) \mathcal{H}\left(Y_{\text{true}}, P_{\text{student}}^1\right)$$

where $\alpha = 0.60$, $\tau = 2.0$, and $D_{\text{KL}}(P \parallel Q) = \sum_{c=1}^C P(c) \ln\frac{P(c)}{Q(c)}$.

### 2.5 Temporal Smoothing & Hysteresis Decision Rule
Instantaneous probabilities $P_{\text{raw}}[k, c]$ are stabilized against transient spurious noise spikes via an Exponential Moving Average (EMA) filter with smoothing factor $\beta = 0.60$:

$$P_{\text{smooth}}[k, c] = \beta \cdot P_{\text{raw}}[k, c] + (1 - \beta) \cdot P_{\text{smooth}}[k-1, c], \quad c \in \{0, 1, 2, 3\}$$

A classification trigger event is asserted if and only if the Root Mean Square (RMS) audio energy satisfies $E_{\text{RMS}} \ge \gamma_{\text{rms}}$ ($0.008$) and the leading target class confidence satisfies:

$$\hat{c} = \arg\max_{c \in \{0, 1, 2\}} P_{\text{smooth}}[k, c], \quad \text{subject to } P_{\text{smooth}}[k, \hat{c}] \ge \theta_{\hat{c}}$$

where class-specific hysteresis thresholds are configured as:
- $\theta_{\text{Dog}} = 0.52$
- $\theta_{\text{DJ}} = 0.52$
- $\theta_{\text{Traffic}} = 0.45$

If thresholds are not met, the system defaults to class index 3 (`Background`).

---

## 3. Static Memory Allocation & Resource Budget

The firmware is designed to operate entirely within internal high-speed SRAM, eliminating PSRAM bus contention and external memory access latency.

### 3.1 SRAM Allocation Breakdown (512 KB Budget)

| Memory Consumer | Description / Buffer Dimension | Allocation Size | SRAM Share | Memory Residency |
| :--- | :--- | :--- | :--- | :--- |
| **Audio Ring Buffer** | $(N_{\text{patch}} + N_{\text{win}}) = 15,760 \times 4\text{ B}$ | $63,040\text{ bytes}$ ($61.6\text{ KB}$) | $12.02\%$ | Internal SRAM (DMA accessible) |
| **Log-Mel Patch Buffer** | $96 \times 40 \times 4\text{ B}$ float matrix | $15,360\text{ bytes}$ ($15.0\text{ KB}$) | $2.93\%$ | Internal SRAM (.bss) |
| **Mel Extractor State** | FFT real/imag vectors ($2 \times 512 \times 4\text{ B}$) + Power ($257 \times 4\text{ B}$) | $5,124\text{ bytes}$ ($5.0\text{ KB}$) | $0.98\%$ | Internal SRAM (.bss) |
| **I2S DMA Descriptors** | 8 descriptors $\times 512\text{ samples} \times 2\text{ B}$ PCM16 | $8,192\text{ bytes}$ ($8.0\text{ KB}$) | $1.56\%$ | Internal SRAM (DMA direct) |
| **FreeRTOS Task Stack** | Audio inference task stack high-water allocation | $16,384\text{ bytes}$ ($16.0\text{ KB}$) | $3.13\%$ | Internal SRAM |
| **TFLite Micro Arena** | Working activation arena for INT8 DS-CNN tensor operations | $98,304\text{ bytes}$ ($96.0\text{ KB}$) | $18.75\%$ | Internal SRAM (.bss) |
| **Classifier State** | EMA state vectors and object context | $20\text{ bytes}$ ($<0.1\text{ KB}$) | $<0.01\%$ | Internal SRAM |
| **Total Active RAM** | **Peak working memory requirement** | **$206,424\text{ bytes}$ ($201.6\text{ KB}$)** | **$39.37\%$** | **Safe internal fit ($>317\text{ KB}$ free)** |

### 3.2 External Octal PSRAM & Flash Occupancy

| Storage Medium | Allocated Segment | Byte Footprint | Total Available | Relative Occupancy | Functional Headroom |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Internal SRAM** | Active Stack, Heap, & Static Buffers | $201.6\text{ KB}$ | $512.0\text{ KB}$ | $39.4\%$ | $310.4\text{ KB}$ ($60.6\%$ headroom) |
| **External PSRAM** | Standby Buffer (Optional Arena) | $0.0\text{ MB}$ ($96.0\text{ KB}$ opt.) | $8.0\text{ MB}$ | $0.0\%$ | $8.0\text{ MB}$ ($100\%$ free for Wi-Fi / app) |
| **Flash Partition** | Firmware + FlatBuffer + DSP LUT | $\sim 1.46\text{ MB}$ | $8.0\text{ MB}$ | $18.3\%$ | $6.54\text{ MB}$ ($81.7\%$ headroom) |
| *Flash: Binary* | ESP-IDF Core + FreeRTOS + Drivers | $\sim 1.38\text{ MB}$ | $2.0\text{ MB}$ (Factory) | $69.0\%$ | Partition fit compliant |
| *Flash: Model* | Quantized INT8 DS-CNN FlatBuffer | $75,000\text{ bytes}$ | Flash (.rodata) | $0.9\%$ | Stored directly in read-only segment |
| *Flash: LUT* | Hann Window (400) + Mel Sparse Weights (455) | $8,800\text{ bytes}$ | Flash (.rodata) | $0.1\%$ | Pre-computed static arrays |

---

## 4. Computational Complexity & CPU Cycle Analysis

Theoretical calculations and hardware profiling were conducted for the Xtensa LX7 dual-core processor clocked at $240\text{ MHz}$ ($2.4 \times 10^8\text{ cycles/second}$).

### 4.1 DSP Operation Complexity (per 0.96s Patch)
Each audio patch comprises 96 consecutive frames:
1. **Periodic Hann Windowing:**  
   $$96 \times 400 = 38,400\text{ operations}$$
2. **512-Point Radix-2 FFT:**  
   $$96 \times (9\text{ stages} \times 256\text{ butterflies} \times 10\text{ ops}) = 96 \times 23,040 = 2,211,840\text{ operations}$$
3. **Power Spectrum Calculation:**  
   $$96 \times (257\text{ bins} \times 3\text{ ops}) = 96 \times 771 = 74,016\text{ operations}$$
4. **Sparse Triangular Filterbank Dot-Products:**  
   $$96 \times 455\text{ sparse entries} \times 2\text{ ops (MAC)} = 87,360\text{ operations}$$
5. **Logarithmic Energy Transform:**  
   $$96 \times 40 = 3,840\text{ math calls}$$
- **Total DSP Complexity:** $\approx 2.43\text{ MFLOPs}$.  
- **DSP Cycle Consumption:** $\approx 3,350,000\text{ cycles}$ ($\approx 13.96\text{ ms}$ on Xtensa single-precision FPU).

### 4.2 Inference Complexity
- **Linear INT8 Spectral Student:**  
  $80\text{ features} \times 4\text{ classes} = 320\text{ MACs} \implies \approx 12,000\text{ cycles}$ ($<0.1\text{ ms}$).
- **DS-CNN Student Architecture:**  
  $6.2\text{ million operations} \implies \approx 6,240,000\text{ cycles}$ ($\approx 26.0\text{ ms}$ with ESP-NN SIMD vector acceleration).

### 4.3 Real-Time Timing Budget vs. 240 ms Hop Interval

| Metric | Linear INT8 Student Mode | DS-CNN Quantized Student Mode | Allowable System Limit |
| :--- | :--- | :--- | :--- |
| **Inference Hop Interval (Stride)** | $3,840\text{ samples}$ ($240.0\text{ ms}$) | $3,840\text{ samples}$ ($240.0\text{ ms}$) | $240.0\text{ ms}$ ($4.167\text{ Hz}$) |
| **Available Clock Cycles per Hop** | $57,600,000\text{ cycles}$ | $57,600,000\text{ cycles}$ | $57,600,000\text{ cycles}$ |
| **DSP Feature Extraction Time** | $13.9\text{ ms}$ ($3.35\text{M cycles}$) | $13.9\text{ ms}$ ($3.35\text{M cycles}$) | — |
| **Model Classification Time** | $<0.1\text{ ms}$ ($0.01\text{M cycles}$) | $26.0\text{ ms}$ ($6.24\text{M cycles}$) | — |
| **Total Task Execution Latency** | **$14.0\text{ ms}$** | **$39.9\text{ ms}$** | **$240.0\text{ ms}$ max** |
| **Core 1 Active Processing Load** | **$5.83\%$** | **$16.63\%$** | **$\le 100\%$** |
| **Core 1 Remaining Idle Headroom** | **$94.17\%$** | **$83.37\%$** | Margin $> 83\%$ |
| **Core 0 Processing Load** | **$0.0\%$** | **$0.0\%$** | Reserved for Wi-Fi / OS |

---

## 5. Hardware Pinout & Physical Interface Specification

The ESP32-S3 communicates with the digital MEMS microphone via standard Philips I2S protocol and asserts three discrete high-efficiency status LEDs for physical classification indication.

### 5.1 Hardware Pin Assignment Table

| Subsystem | Signal Function | Module Pin | ESP32-S3 GPIO | Electrical Configuration / Logic Level |
| :--- | :--- | :--- | :--- | :--- |
| **I2S MEMS Mic** (INMP441) | Serial Bit Clock (`SCK` / `BCLK`) | `SCK` | **GPIO 4** | Standard CMOS Output |
| **I2S MEMS Mic** | Word Select (`WS` / `LRCLK`) | `WS` | **GPIO 5** | 16 kHz Frame Synchronization Clock |
| **I2S MEMS Mic** | Serial Data Out (`SD` / `DOUT`) | `SD` | **GPIO 6** | Digital Audio Input (Internal Pull-down) |
| **I2S MEMS Mic** | Channel Selection (`L/R`) | `L/R` | **GND** | Tied to GND (Left Audio Channel Select) |
| **I2S MEMS Mic** | Power & Ground | `VDD`, `GND` | **3.3V, GND** | Regulated 3.3V (Strict: 5V causes destruction) |
| **Status LED: Dog** | Class 0 Indicator (Red) | Anode | **GPIO 15** | Push-Pull Output via $220\,\Omega$ resistor ($1000\text{ ms}$ hold) |
| **Status LED: Traffic**| Class 2 Indicator (Yellow) | Anode | **GPIO 16** | Push-Pull Output via $220\,\Omega$ resistor ($1000\text{ ms}$ hold) |
| **Status LED: DJ** | Class 1 Indicator (Purple/Blue)| Anode | **GPIO 17** | Push-Pull Output via $220\,\Omega$ resistor ($1000\text{ ms}$ hold) |
| **Telemetry UART** | USB Serial / CDC Console | `TXD`, `RXD` | **USB Port** | 115200 Baud, 8N1 serial protocol |

*Note: All hardware pin allocations are defined as preprocessor macros in `esp32s3_firmware/main/config.h` and can be adjusted to match custom board revisions.*

---

## 6. Firmware Compilation & Deployment

The codebase provides dual build support for industrial ESP-IDF toolchains and standard Arduino IDE development environments.

### 6.1 ESP-IDF v5.x Compilation (Recommended Production Target)

Requires ESP-IDF version 5.0 or later with the Espressif Component Manager.

```bash
cd esp32s3_firmware

# 1. Set the build target to ESP32-S3
idf.py set-target esp32s3

# 2. Compile firmware binary, bootloader, and partition table
idf.py build

# 3. Flash to target board and monitor serial output (substitute appropriate /dev/tty node)
idf.py -p /dev/ttyACM0 flash monitor
```

The ESP-IDF build system automatically:
- Ingests `sdkconfig.defaults` (configuring 240 MHz CPU, 8 MB Flash, and custom partitions).
- Ingests `idf_component.yml` to resolve `espressif/esp-tflite-micro` and `espressif/esp-dsp`.
- Compiles standard ESP-IDF v5 I2S drivers (`driver/i2s_std.h`).

### 6.2 Arduino Framework Compilation

1. Launch Arduino IDE (version 2.0 or higher).
2. Install the **esp32** board package by Espressif Systems (Tools -> Board Manager).
3. Open `esp32s3_firmware/arduino_esp32s3_sketch/esp32s3_audio_classifier.ino`.
4. Configure board parameters under **Tools**:
   - **Board**: `"ESP32S3 Dev Module"`
   - **CPU Frequency**: `"240MHz (WiFi)"`
   - **Flash Size**: `"8MB (64Mb)"`
   - **PSRAM**: `"OPI PSRAM"` (if N8R8 module) or `"Disabled"` (if N8 module)
   - **Upload Mode**: `"UART0 / Hardware CDC"`
5. Connect target device via USB and click **Upload**.

---

## 7. Host Simulation & Verification Harness

To enable deterministic verification of the C++ signal processing chain and INT8 classification logic on standard host workstations (without requiring an attached physical ESP32-S3), a self-contained simulation testbench is provided.

### 7.1 Compilation and Execution

```bash
cd simulation
make run
```

### 7.2 Verification Operations
The simulation harness compiles with standard C++17 (`g++`) and executes:
1. **Memory Budgeting Audit:** Validates struct sizes, buffer lengths, and SRAM occupancy against the 512 KB ESP32-S3 limit.
2. **Cycle Timing Simulation:** Calculates FLOP complexity and validates the 240 ms real-time hop deadline.
3. **Bit-Exact Audio Validation:** Parses four 16 kHz 16-bit PCM test files (`dog_bark.wav`, `dj_electronic.wav`, `traffic_street.wav`, `background_silence.wav`), feeds them through the exact DSP extractor and classifier routines, and prints confidence scores and simulated GPIO status.

---

## 8. Telemetry & Communications Protocol

Real-time telemetry frames are transmitted over USB-UART (115200 baud) and streaming WebSockets as single-line JSON payloads at 4.167 Hz.

### 8.1 JSON Schema Specification (Draft-07)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ESP32S3AudioTelemetryPayload",
  "type": "object",
  "properties": {
    "label": {
      "type": "string",
      "enum": ["Dog", "DJ", "Traffic", "Background"],
      "description": "Winning classification category label"
    },
    "triggered": {
      "type": "boolean",
      "description": "Hardware assertion state (true if winning class exceeds confidence and RMS gates)"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Smoothed probability associated with the winning class"
    },
    "probs": {
      "type": "array",
      "items": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
      "minItems": 4,
      "maxItems": 4,
      "description": "Smoothed probability vector: [Dog, DJ, Traffic, Background]"
    },
    "rms": {
      "type": "number",
      "minimum": 0.0,
      "description": "Root Mean Square acoustic energy across the active hop buffer"
    },
    "latency_ms": {
      "type": "integer",
      "minimum": 0,
      "description": "Combined DSP feature extraction and neural inference execution time in milliseconds"
    },
    "leds": {
      "type": "array",
      "items": { "type": "integer", "enum": [0, 1] },
      "minItems": 3,
      "maxItems": 3,
      "description": "Discrete logic levels of status indicators: [GPIO 15 (Dog), GPIO 16 (Traffic), GPIO 17 (DJ)]"
    }
  },
  "required": ["label", "triggered", "confidence", "probs", "rms", "latency_ms", "leds"]
}
```

### 8.2 Sample Transmission Payload
```json
{"label":"Traffic","triggered":true,"confidence":0.502,"probs":[0.135,0.240,0.502,0.123],"rms":0.3272,"latency_ms":14,"leds":[0,1,0]}
```

---

## 9. Machine Learning Training & Model Distillation Pipeline

Retraining or customizing acoustic categories is supported through the Python pipeline in `ml_pipeline/`.

### 9.1 Local Execution

```bash
cd ml_pipeline

# 1. Install prerequisites
pip install -r requirements.txt

# 2. Execute knowledge distillation from YAMNet teacher into INT8 student DS-CNN
python3 distill_from_yamnet.py --data-dir dataset --epochs 35

# 3. Verify quantized TFLite model inference
python3 verify_int8_model.py
```

The script automatically generates `model_int8.tflite` and serializes the quantized C array directly into `esp32s3_firmware/main/model_data.cc`.

### 9.2 Interactive Cloud Training
For cloud environments lacking local GPU hardware, `ml_pipeline/ESP32S3_YAMNet_Distillation_Colab.ipynb` can be executed within Google Colab or Jupyter Lab. The notebook ingests the ESC-50 dataset, extracts Log-Mel patches, trains the student network, performs Post-Training Quantization (PTQ), and yields updated C headers.

---

## 10. Web-Based Telemetry & Visualization Interface

The `web_dashboard/` directory provides a client-side interface for telemetry monitoring, acoustic verification, and hardware emulation.

```bash
cd web_dashboard
python3 -m http.server 8080
```
Open `http://localhost:8080` in a browser supporting the WebSerial API (e.g., Google Chrome or Microsoft Edge):
- **Serial Telemetry:** Select **Connect ESP32-S3 (USB Serial)** to establish a 115200 baud direct link with the hardware.
- **WebSocket Streaming:** Connect to `ws://<device-ip>/ws` for wireless telemetry over local Wi-Fi.
- **Interactive Verification:** Selecting target category visualizer cards exercises status animations and emulates GPIO transitions.
