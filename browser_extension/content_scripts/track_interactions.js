// Tracks video seeking/interaction state and exposes it via
// window.__videoInteractionState, which get_video_bounds.js reads on each
// capture tick.  Handles video elements added dynamically after page load.
(function () {
  // Bail out if already injected (e.g. due to frame navigation).
  if (window.__videoInteractionState) return;

  function getLargestVideo() {
    const videos = Array.from(document.querySelectorAll('video')).filter(
      v => v.videoWidth > 0 && v.videoHeight > 0
    );

    if (videos.length === 0) return null;

    // Pick the largest video by rendered area
    const video = videos.reduce((best, v) => {
      const r = v.getBoundingClientRect();
      const bestR = best.getBoundingClientRect();
      return r.width * r.height > bestR.width * bestR.height ? v : best;
    });

    return video;
  }

  const state = (window.__videoInteractionState = {
    videoElement: getLargestVideo(),
    isSeeking: false,
    lastSeekMs: 0,   // epoch ms of the most recent seeking/seeked event
  });

  function initTracking(video) {
    video.addEventListener('seeking', () => {
      console.debug('Video seeking started');
      state.isSeeking = true;
      state.lastSeekMs = Date.now();
    });

    video.addEventListener('seeked', () => {
      console.debug('Video seeking ended');
      state.isSeeking = false;
      state.lastSeekMs = Date.now();
    });

    video.addEventListener('unload', () => {
      // If the tracked video is removed from the DOM, clear it from state so
      // get_video_bounds.js doesn't try to read properties from a detached element.
      if (state.videoElement === video) {
        state.videoElement = null;
      }
    });
  }

  function attachTo(video) {
    // check if the video is larger than the currently tracked one
    if (state.videoElement) {
      const r = video.getBoundingClientRect();
      const sr = state.videoElement.getBoundingClientRect();
      if (r.width * r.height <= sr.width * sr.height) {
        return;
      }
    }

    console.debug('Tracking video interactions on', video);
    state.videoElement = video;
    initTracking(video);
  }

  // Attach to any video elements already in the DOM.
  document.querySelectorAll('video').forEach(attachTo);

  // Watch for videos added dynamically (SPAs, deferred loads, etc.).
  const observer = new MutationObserver(mutations => {
    for (const mut of mutations) {
      for (const node of mut.addedNodes) {
        if (node.nodeName === 'VIDEO') {
          attachTo(node);
        } else if (node.querySelectorAll) {
          node.querySelectorAll('video').forEach(attachTo);
        }
      }
    }
  });

  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
