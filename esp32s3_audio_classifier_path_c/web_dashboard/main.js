import { initModel } from './ml.js';
import { startMicrophone, processUploadedAudioFile } from './audio.js';
import { triggerGlow, updateHardwareTelemetry } from './ui.js';

const loadingScreen = document.getElementById('loading-screen');
const mainApp = document.getElementById('main-app');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const statusPill = document.getElementById('connection-status');
const statusSubtitle = document.getElementById('status-subtitle');

// 1. Connect to ESP32-S3 over USB Serial (WebSerial API - Chrome / Edge)
document.getElementById('connect-serial').addEventListener('click', async () => {
    if (!('serial' in navigator)) {
        alert('WebSerial requires Google Chrome or Microsoft Edge over HTTPS or localhost.');
        return;
    }
    try {
        const port = await navigator.serial.requestPort();
        await port.open({ baudRate: 115200 });
        statusPill.innerText = 'ESP32-S3 Connected (USB Serial 115200)';
        statusPill.classList.add('connected');
        statusSubtitle.innerText = 'Streaming real-time INT8 Student CNN detections + GPIO LED states from ESP32-S3...';

        const textDecoder = new TextDecoderStream();
        port.readable.pipeTo(textDecoder.writable);
        const reader = textDecoder.readable.getReader();
        let lineBuffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            lineBuffer += value;
            const lines = lineBuffer.split('\n');
            lineBuffer = lines.pop();
            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
                    try {
                        const payload = JSON.parse(trimmed);
                        updateHardwareTelemetry(payload);
                    } catch (_) {}
                }
            }
        }
    } catch (err) {
        statusSubtitle.innerText = `USB Serial connection cancelled or closed: ${err.message}`;
    }
});

// 2. Connect to ESP32-S3 over Wi-Fi WebSocket
document.getElementById('connect-ws').addEventListener('click', () => {
    const defaultHost = window.location.hostname || '192.168.4.1';
    const ip = prompt('Enter ESP32-S3 IP address or hostname:', defaultHost);
    if (!ip) return;
    const ws = new WebSocket(`ws://${ip}/ws`);
    ws.onopen = () => {
        statusPill.innerText = `ESP32-S3 Connected (ws://${ip}/ws)`;
        statusPill.classList.add('connected');
        statusSubtitle.innerText = 'Receiving live Wi-Fi WebSocket detections from ESP32-S3...';
    };
    ws.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            updateHardwareTelemetry(payload);
        } catch (_) {}
    };
    ws.onerror = () => {
        statusSubtitle.innerText = `Could not reach ws://${ip}/ws. Check ESP32-S3 IP or use USB Serial.`;
    };
});

// 3. Browser Microphone Mode (Lazy-loads TF.js YAMNet Teacher only when clicked)
document.getElementById('start-mic').addEventListener('click', async () => {
    loadingScreen.classList.remove('hidden');
    mainApp.classList.add('hidden');
    await initModel((fraction) => {
        const percentage = Math.floor(fraction * 100);
        progressBar.style.width = `${percentage}%`;
        progressText.innerText = `${percentage}%`;
    });
    loadingScreen.classList.add('hidden');
    mainApp.classList.remove('hidden');
    await startMicrophone();
    statusPill.innerText = 'Browser Microphone Active (YAMNet Teacher)';
    statusPill.classList.add('connected');
    statusSubtitle.innerText = 'Listening via browser microphone using YAMNet Teacher...';
});

// 4. Upload Audio File Mode
document.getElementById('audio-upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    loadingScreen.classList.remove('hidden');
    mainApp.classList.add('hidden');
    await initModel((fraction) => {
        const percentage = Math.floor(fraction * 100);
        progressBar.style.width = `${percentage}%`;
        progressText.innerText = `${percentage}%`;
    });
    loadingScreen.classList.add('hidden');
    mainApp.classList.remove('hidden');
    statusPill.innerText = `Playing: ${file.name}`;
    statusPill.classList.add('connected');
    await processUploadedAudioFile(file);
});

// 5. Manual telemetry simulation event handlers
document.querySelectorAll('.image-wrapper').forEach((card) => {
    card.addEventListener('click', () => {
        const label = card.getAttribute('data-sim');
        const simProbs = {
            'Dog': [0.91, 0.03, 0.04, 0.02],
            'Traffic': [0.04, 0.03, 0.89, 0.04],
            'DJ': [0.02, 0.93, 0.03, 0.02]
        }[label];
        updateHardwareTelemetry({
            label,
            triggered: true,
            confidence: Math.max(...simProbs),
            probs: simProbs,
            rms: 0.142,
            latency_ms: 24
        });
    });
});
