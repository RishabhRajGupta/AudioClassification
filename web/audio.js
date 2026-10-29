import { updateBuffer } from './ml.js';

export async function startMicrophone(){
    const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const audioEngine = new window.AudioContext({ sampleRate: 16000 });
    console.log("Audio stream: " + audioStream);
    console.log("Audio Engine: " + audioEngine);
    
    const audio = audioEngine.createMediaStreamSource(audioStream);
    const chunkingProcessor = audioEngine.createScriptProcessor(4096, 1, 1);
    audio.connect(chunkingProcessor);
    chunkingProcessor.connect(audioEngine.destination);
    
    chunkingProcessor.onaudioprocess = function(event) {
        const newChunk = event.inputBuffer.getChannelData(0);
        updateBuffer(newChunk);
    }
}