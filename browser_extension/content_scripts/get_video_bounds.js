// Find the largest visible video element on the page and return its playing
// status, interaction state, and bounding rect. Returns null if no video is
// found.
//
// Seeking state comes from two sources:
//   - video.seeking  — true while the browser is fetching the new position
//   - window.__videoInteractionState — set by track_interactions.js; also
//     exposes lastSeekMs so callers can detect *recently* completed seeks
//     even after video.seeking has returned to false.
(function () {
  const clamp0 = n => Math.max(0, n);

  // How long after a seek completes to still consider the user "interacting".
  const RECENT_SEEK_WINDOW_MS = 5000;

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

  // Read shared interaction state (maintained by track_interactions.js).
  const iState = window.__videoInteractionState;
  const seeking = iState?.isSeeking || video.seeking;
  const recentlySeeked =
    !seeking &&
    iState != null &&
    Date.now() - iState.lastSeekMs < RECENT_SEEK_WINDOW_MS;

  // Check if the video is paused or ended
  if (video.paused || video.ended) {
    return { playing: false, seeking, recentlySeeked };
  }

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  return {
    playing: true,
    seeking,
    recentlySeeked,
    x:      Math.round(clamp0(rect.left   * dpr)),
    y:      Math.round(clamp0(rect.top    * dpr)),
    width:  Math.round(clamp0(rect.width  * dpr)),
    height: Math.round(clamp0(rect.height * dpr)),
  };
})();
