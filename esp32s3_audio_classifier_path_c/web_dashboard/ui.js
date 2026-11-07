const timers = {
    'Dog': null,
    'Traffic': null,
    'DJ': null
};

export function triggerGlow(classifiedName, confidence = null) {
    const image = document.getElementById(classifiedName);
    const ledIndicator = document.getElementById(`led-${classifiedName}`);
    const probSpan = document.getElementById(`prob-${classifiedName}`);
    const wrapper = image ? image.closest('.image-wrapper') : null;
    const fallback = document.getElementById(`${classifiedName}-fallback`);
    if (!image && !wrapper) return;

    if (image) image.classList.add('active-glow');
    if (fallback) fallback.classList.add('active-glow');
    if (wrapper) wrapper.classList.add('active-card-glow', `active-glow-${classifiedName}`);
    if (ledIndicator) ledIndicator.classList.add('led-on');
    if (probSpan && confidence !== null) {
        probSpan.innerText = `(${Math.round(confidence * 100)}%)`;
    }

    clearTimeout(timers[classifiedName]);
    timers[classifiedName] = setTimeout(() => {
        if (image) image.classList.remove('active-glow');
        if (fallback) fallback.classList.remove('active-glow');
        if (wrapper) wrapper.classList.remove('active-card-glow', `active-glow-${classifiedName}`);
        if (ledIndicator) ledIndicator.classList.remove('led-on');
    }, 1000);
}

export function updateHardwareTelemetry(payload) {
    if (!payload) return;
    const tel = document.getElementById('telemetry-text');
    if (tel && payload.latency_ms !== undefined && payload.rms !== undefined) {
        tel.innerHTML = `Latency: <strong>${payload.latency_ms} ms</strong> &nbsp;|&nbsp; RMS: <strong>${Number(payload.rms).toFixed(3)}</strong>`;
    }
    if (Array.isArray(payload.probs) && payload.probs.length >= 3) {
        const names = ['Dog', 'DJ', 'Traffic'];
        names.forEach((n, idx) => {
            const el = document.getElementById(`prob-${n}`);
            if (el) el.innerText = `${Math.round(payload.probs[idx] * 100)}%`;
        });
    }
    if (payload.triggered && payload.label && payload.label !== 'Background') {
        triggerGlow(payload.label, payload.confidence);
    }
}

window.addEventListener('soundDetected', (event) => {
    const detail = event.detail;
    if (typeof detail === 'string') {
        triggerGlow(detail);
    } else if (detail && detail.label) {
        triggerGlow(detail.label, detail.confidence);
    }
});
