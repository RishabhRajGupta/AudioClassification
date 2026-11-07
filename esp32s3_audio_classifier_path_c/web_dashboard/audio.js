import { updateBuffer } from './ml.js';

let audioCtx = null;

export async function startMicrophone() {
    if (!audioCtx) {
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        } catch (_) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
    }
    if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const source = audioCtx.createMediaStreamSource(stream);
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    const muteGain = audioCtx.createGain();
    muteGain.gain.value = 0; // Prevent microphone feedback into speakers

    source.connect(processor);
    processor.connect(muteGain);
    muteGain.connect(audioCtx.destination);

    processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        updateBuffer(new Float32Array(inputData));
    };
}

export async function processUploadedAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    let tempCtx;
    try {
        tempCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    } catch (_) {
        tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);

    // Stream in 4096-sample hops every 200ms to animate the UI
    let offset = 0;
    const hop = 4096;
    const timer = setInterval(() => {
        if (offset + hop > channelData.length) {
            clearInterval(timer);
            return;
        }
        updateBuffer(channelData.slice(offset, offset + hop));
        offset += hop;
    }, 200);
}
