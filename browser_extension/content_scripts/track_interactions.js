// Tracks video seeking/interaction state and exposes it via
// window.__videoInteractionState, which get_video_bounds.js reads on each
// capture tick.  Handles video elements added dynamically after page load.
(function () {
  // Bail out if already injected (e.g. due to frame navigation).
  if (window.__videoInteractionState) return;

  const state = (window.__videoInteractionState = {
    isSeeking: false,
    lastSeekMs: 0,   // epoch ms of the most recent seeking/seeked event
  });

  function attachTo(video) {
    if (video.__interactionTracked) return;
    video.__interactionTracked = true;

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
