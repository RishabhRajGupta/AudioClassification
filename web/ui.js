const timers = {
    'Dog': null,
    'Traffic': null,
    'DJ': null
};

function triggerGlow(classifiedName){
    const image = document.getElementById(classifiedName);
    image.classList.add('active-glow');
    clearTimeout(timers[classifiedName]);

    timers[classifiedName] = setTimeout(()=>{
        image.classList.remove('active-glow');
    }, 1000);
}


window.addEventListener('soundDetected', (event) => {
    triggerGlow(event.detail);
})