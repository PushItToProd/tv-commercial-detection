// Find the largest visible video element on the page and return its playing
// status, interaction state, timebase, and bounding rect. Returns null if no
// video is found.
//
// Seeking state comes from two sources:
//   - video.seeking  — true while the browser is fetching the new position
//   - window.__videoInteractionState — set by track_interactions.js; also
//     exposes lastSeekMs so callers can detect *recently* completed seeks
//     even after video.seeking has returned to false.
(function () {
  const clamp0 = n => Math.max(0, n);

  // How long after a seek completes to still consider the user "interacting".
  const RECENT_SEEK_WINDOW_MS = 3000;

  const iState = window.__videoInteractionState;
  const video = iState?.videoElement;
  if (!video) return null;

  const videoTitle = document.querySelector('.ypc-video-title-text')?.textContent ?? null;
  const networkName = document.querySelector('.ypc-network-logo')?.textContent.trim() ?? null;

  // A stable identifier for the program, which the title is not — the title is
  // briefly empty while the player navigates. Covers the `/watch/<id>` and
  // `?v=<id>` shapes; anything else reports null rather than guessing at a
  // path segment that might mean something else entirely.
  function videoId() {
    try {
      const url = new URL(location.href);
      const v = url.searchParams.get('v');
      if (v) return v;
      const parts = url.pathname.split('/').filter(Boolean);
      for (const marker of ['watch', 'video']) {
        const i = parts.indexOf(marker);
        if (i >= 0 && parts[i + 1]) return parts[i + 1];
      }
    } catch (e) {
      // A page URL we can't parse just means no id.
    }
    return null;
  }

  // Whether `currentTime` is a position in the program or time since the player
  // loaded. A live stream reports an infinite duration and a DVR window whose
  // start creeps forward; a recording reports a finite duration and a range
  // starting at 0. Only the former makes offsets comparable across separate
  // capture passes, so it is worth knowing per frame rather than assumed.
  //
  // `duration` is deliberately passed through as-is: Infinity and NaN are the
  // informative cases, and the server maps them to `is_live` true and unknown.
  // Reading start/end on an empty TimeRanges throws, hence the length guard;
  // in the rare multi-range case the outermost bounds are the useful summary.
  function timebase(video) {
    const seekable = video.seekable;
    const n = seekable ? seekable.length : 0;
    return {
      videoId: videoId(),
      duration: video.duration,
      seekableStart: n > 0 ? seekable.start(0) : null,
      seekableEnd: n > 0 ? seekable.end(n - 1) : null,
    };
  }

  // Read shared interaction state (maintained by track_interactions.js).
  const seeking = iState?.isSeeking || video.seeking;
  const recentlySeeked =
    !seeking &&
    iState != null &&
    Date.now() - iState.lastSeekMs < RECENT_SEEK_WINDOW_MS;

  // Check if the video is paused or ended
  if (video.paused || video.ended) {
    return {
      playing: false,
      seeking,
      recentlySeeked,
      videoTitle,
      networkName,
      currentTime: video.currentTime,
      ...timebase(video),
    };
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
    videoTitle,
    networkName,
    currentTime: video.currentTime,
    ...timebase(video),
  };
})();
