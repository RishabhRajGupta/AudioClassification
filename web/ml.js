import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs/+esm';

const MODEL_URL = 'https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1';
let yamnetModel;

const BUFFER_SIZE=15600;
const slidingBuffer = new Float32Array(BUFFER_SIZE);

const TARGET_MAP = {
  // --- DOG SOUNDS ---
  67: 'Dog',   // Animal
  68: 'Dog',   // Domestic animals, pets
  69: 'Dog',   // Dog
  70: 'Dog',   // Bark
  71: 'Dog',   // Yip
  72: 'Dog',   // Howl
  73: 'Dog',   // Bow-wow
  74: 'Dog',   // Growling
  75: 'Dog',   // Whimper (dog)
  117: 'Dog',  // Canidae, dogs, wolves

  // --- TRAFFIC SOUNDS ---
  294: 'Traffic', // Vehicle
  300: 'Traffic', // Motor vehicle (road)
  301: 'Traffic', // Car
  302: 'Traffic', // Vehicle horn, car horn, honking
  308: 'Traffic', // Car passing by
  310: 'Traffic', // Truck
  315: 'Traffic', // Bus
  316: 'Traffic', // Emergency vehicle
  317: 'Traffic', // Police car (siren)
  318: 'Traffic', // Ambulance (siren)
  320: 'Traffic', // Motorcycle
  321: 'Traffic', // Traffic noise, roadway noise
  390: 'Traffic', // Siren
  503: 'Traffic', // Outside, urban or manmade
  508: 'Traffic', // Environmental noise

  // --- DJ SOUNDS ---
  132: 'DJ', // Music
  153: 'DJ', // Synthesizer
  154: 'DJ', // Sampler
  158: 'DJ', // Drum machine
  210: 'DJ', // Scratching (performance technique)
  234: 'DJ', // Electronic music
  235: 'DJ', // House music
  236: 'DJ', // Techno
  237: 'DJ', // Dubstep
  238: 'DJ', // Drum and bass
  239: 'DJ', // Electronica
  240: 'DJ', // Electronic dance music
  242: 'DJ', // Trance music
  269: 'DJ', // Dance music
  457: 'DJ'  // Effects unit
};

const THRESHOLDS = {
  'Dog': 0.5,
  'DJ': 0.5,
  'Traffic': 0.1
};

export function updateBuffer(newChunk){
    slidingBuffer.set(slidingBuffer.subarray(newChunk.length, BUFFER_SIZE), 0);
    slidingBuffer.set(newChunk, BUFFER_SIZE - newChunk.length);
    runInference();
}

export async function initModel(progressCallback){
    yamnetModel = await tf.loadGraphModel(MODEL_URL, { 
        fromTFHub:true,
        onProgress: progressCallback
    });
}

function runInference(){
    if(yamnetModel!=null){
        tf.tidy(() => {
            const audioTensor = tf.tensor1d(slidingBuffer);
            const [scores] = yamnetModel.predict(audioTensor);
            const predictedArray = scores.dataSync();

            Object.keys(TARGET_MAP).forEach((key) => {
                const confidenceScore = predictedArray[key];
                const label = TARGET_MAP[key];
                const requiredThreshold = THRESHOLDS[label];
                
                if(confidenceScore > requiredThreshold){
                    window.dispatchEvent(new CustomEvent('soundDetected', {detail: label}));
                }
            })
        })
    }
}