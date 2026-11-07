import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/+esm';

const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1';
let yamnetModel = null;

const BUFFER_SIZE = 15600;
const slidingBuffer = new Float32Array(BUFFER_SIZE);

const TARGET_MAP = {
  // --- DOG SOUNDS ---
  67: 'Dog', 68: 'Dog', 69: 'Dog', 70: 'Dog', 71: 'Dog',
  72: 'Dog', 73: 'Dog', 74: 'Dog', 75: 'Dog', 117: 'Dog',
  // --- TRAFFIC SOUNDS ---
  294: 'Traffic', 300: 'Traffic', 301: 'Traffic', 302: 'Traffic', 308: 'Traffic',
  310: 'Traffic', 315: 'Traffic', 316: 'Traffic', 317: 'Traffic', 318: 'Traffic',
  320: 'Traffic', 321: 'Traffic', 390: 'Traffic', 503: 'Traffic', 508: 'Traffic',
  // --- DJ SOUNDS ---
  132: 'DJ', 153: 'DJ', 154: 'DJ', 158: 'DJ', 210: 'DJ',
  234: 'DJ', 235: 'DJ', 236: 'DJ', 237: 'DJ', 238: 'DJ',
  239: 'DJ', 240: 'DJ', 242: 'DJ', 269: 'DJ', 457: 'DJ'
};

const THRESHOLDS = {
  'Dog': 0.5,
  'DJ': 0.5,
  'Traffic': 0.1
};

export function updateBuffer(newChunk) {
    slidingBuffer.set(slidingBuffer.subarray(newChunk.length, BUFFER_SIZE), 0);
    slidingBuffer.set(newChunk, BUFFER_SIZE - newChunk.length);
    runInference();
}

export async function initModel(progressCallback) {
    if (yamnetModel) return;
    yamnetModel = await tf.loadGraphModel(MODEL_URL, {
        fromTFHub: true,
        onProgress: progressCallback
    });
}

function runInference() {
    if (yamnetModel != null) {
        tf.tidy(() => {
            const audioTensor = tf.tensor1d(slidingBuffer);
            const [scores] = yamnetModel.predict(audioTensor);
            const predictedArray = scores.dataSync();

            Object.keys(TARGET_MAP).forEach((key) => {
                const confidenceScore = predictedArray[key];
                const label = TARGET_MAP[key];
                const requiredThreshold = THRESHOLDS[label];
                if (confidenceScore > requiredThreshold) {
                    window.dispatchEvent(new CustomEvent('soundDetected', {
                        detail: { label, confidence: confidenceScore }
                    }));
                }
            });
        });
    }
}
