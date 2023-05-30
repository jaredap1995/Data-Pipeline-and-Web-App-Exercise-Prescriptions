const video = window.parent.document.getElementById('video');
        const audio = window.parent.document.getElementById('myaudio');

        video.addEventListener('play', () => {
            audio.currentTime = video.currentTime;
            audio.play();
        });

        video.addEventListener('pause', () => {
            audio.pause();
        });

        video.addEventListener('seeked', () => {
            audio.currentTime = video.currentTime;
        });

        video.addEventListener('ended', () => {
            audio.pause();
        });