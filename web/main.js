import { initModel } from './ml.js';
import { startMicrophone } from './audio.js';
import './ui.js';

const loadingScreen = document.getElementById('loading-screen');
const mainApp = document.getElementById('main-app');
const progressBar = document.getElementById('progress-bar');
const progressText = document.getElementById('progress-text');
const micButton = document.getElementById('start-mic');

await initModel((fraction) => {
    const percentage = Math.floor(fraction*100);
    progressBar.style.width = `${percentage}%`;
    progressText.innerText = `${percentage}%`;
});

loadingScreen.classList.add('hidden');
mainApp.classList.remove('hidden');

const button = document.getElementById('start-mic');
button.addEventListener('click', () => {
    startMicrophone();
    document.querySelector('header p').innerText = "Microphone active. Listening for audio...";
})