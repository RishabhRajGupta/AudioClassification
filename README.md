# Real-Time Audio Event Classification System

This repository contains two independent implementations for real-time acoustic event classification (**Dog**, **DJ / Electronic Music**, and **Traffic**):

1. **Browser Implementation (`web/`)**: Runs fully inside the web browser using TensorFlow.js and Google's pre-trained YAMNet model.
2. **Embedded Hardware Implementation (`esp32s3_audio_classifier_path_c/`)**: Runs on an ESP32-S3 microcontroller using a lightweight INT8 student CNN, I2S microphone, and physical GPIO LEDs.

---

## Comparison of the Two Implementations

| Feature | 1. Web Version (`web/`) | 2. Embedded Version (`esp32s3_audio_classifier_path_c/`) |
|---|---|---|
| **Target Platform** | Web Browser (Chrome, Edge, Firefox) | ESP32-S3 Microcontroller |
| **Model** | Google YAMNet (Full Float32, ~15 MB) | Distilled Student CNN (Quantized INT8, ~75 KB) |
| **Inference Engine** | TensorFlow.js Graph Model | TFLite Micro / Optimized C++ Matrix Math |
| **Audio Input** | Laptop / PC Microphone (Web Audio API) | INMP441 I2S MEMS Microphone (16 kHz mono) |
| **Output / Feedback** | Browser UI with glowing card animations | 3 Physical GPIO LEDs (GPIO 15, 16, 17) + Web Dashboard |
| **Internet Required** | Yes (fetches model weights on first load) | No (runs 100% offline on the chip) |
| **RAM Footprint** | Browser memory (~100 MB) | 201.6 KB (fits inside 512 KB internal SRAM) |

---

## Repository Structure

```
AudioClassification/
├── web/                              # Implementation 1: Browser-based Classifier
│   ├── index.html                    # Web interface
│   ├── main.js                       # Application entry point
│   ├── audio.js                      # Web Audio API microphone handler
│   ├── ml.js                         # TensorFlow.js YAMNet inference
│   ├── ui.js                         # UI glow animations
│   ├── style.css                     # Dark-mode styling
│   └── images/                       # Target class graphics
│
├── esp32s3_audio_classifier_path_c/  # Implementation 2: ESP32-S3 Embedded System
│   ├── esp32s3_firmware/             # ESP-IDF & Arduino firmware for ESP32-S3
│   ├── ml_pipeline/                  # YAMNet distillation, dataset, and INT8 scripts
│   ├── simulation/                   # Host C++ simulator and Makefile
│   ├── web_dashboard/                # Real-time WebSerial / WebSocket dashboard
│   └── README.md                     # Detailed embedded technical specification
│
└── README.md                         # Project overview and navigation
```

---

## Implementation 1: Running the Web Version

The browser version runs directly on your computer without any hardware.

### Steps to Run:

1. Navigate to the `web/` folder:
   ```bash
   cd web
   ```
2. Start a local HTTP server:
   ```bash
   python3 -m http.server 8000
   ```
3. Open `http://localhost:8000` in Google Chrome or Microsoft Edge.
4. Click **Start Microphone** to classify live sounds, or click **Upload Audio** to test a sound file.

---

## Implementation 2: Running the ESP32-S3 Embedded Version

The embedded version is optimized for resource-constrained microcontrollers.

### Hardware Pinout:

| Component | Pin on Component | ESP32-S3 GPIO | Description |
|---|---|---|---|
| **I2S Microphone (INMP441)** | `SCK` / `BCLK` | **GPIO 4** | Bit Clock |
| **I2S Microphone (INMP441)** | `WS` / `LRCLK` | **GPIO 5** | Word Select (16 kHz mono) |
| **I2S Microphone (INMP441)** | `SD` / `DOUT` | **GPIO 6** | Serial Data In |
| **Dog LED** | Anode (`+`) via 220Ω | **GPIO 15** | Red LED for Dog detection |
| **Traffic LED** | Anode (`+`) via 220Ω | **GPIO 16** | Yellow LED for Traffic detection |
| **DJ LED** | Anode (`+`) via 220Ω | **GPIO 17** | Purple/Blue LED for DJ detection |

### 1. Test via Host Simulation (No Hardware Needed):
```bash
cd esp32s3_audio_classifier_path_c/simulation
make run
```

### 2. Build and Flash to ESP32-S3:
```bash
cd esp32s3_audio_classifier_path_c/esp32s3_firmware
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```
*(Or open `arduino_esp32s3_sketch/esp32s3_audio_classifier.ino` in Arduino IDE).*

### 3. Open Real-Time Web Dashboard:
```bash
cd esp32s3_audio_classifier_path_c/web_dashboard
python3 -m http.server 8080
```
Open `http://localhost:8080` in Chrome and click **Connect ESP32-S3 (USB Serial)**.
