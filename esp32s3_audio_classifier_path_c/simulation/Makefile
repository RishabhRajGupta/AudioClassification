# ==============================================================================
# ESP32-S3 Audio Classifier: Host Simulation Harness Makefile
# ==============================================================================
# Compiles the bit-exact C++ DSP feature extraction, quantized student inference,
# memory audit, and cycle budget model against host workstation toolchains.
# ==============================================================================

CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Wpedantic

FIRMWARE_MAIN = ../esp32s3_firmware/main
INCLUDES = -I$(FIRMWARE_MAIN)

SRCS = simulate_esp32s3.cpp \
       $(FIRMWARE_MAIN)/mel_spectrogram.cpp \
       $(FIRMWARE_MAIN)/classifier.cpp \
       $(FIRMWARE_MAIN)/model_data.cc

TARGET = simulate_esp32s3

.PHONY: all run simulate clean help

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

run: $(TARGET)
	./$(TARGET)

simulate: run

clean:
	rm -f $(TARGET) *.o

help:
	@echo "Available targets:"
	@echo "  all       : Build the $(TARGET) executable"
	@echo "  run       : Build and execute the simulation harness"
	@echo "  simulate  : Alias for run"
	@echo "  clean     : Remove generated binaries and object files"
