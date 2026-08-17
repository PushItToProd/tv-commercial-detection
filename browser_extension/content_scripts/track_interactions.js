// Tracks which <video> element is the page's real player, plus its seeking
// state, and exposes both via window.__videoInteractionState, which
// get_video_bounds.js reads on each capture tick.
//
// The element is resolved lazily on every read rather than latched at
// injection time. A page routinely has several <video> elements — a main
// player that hasn't started, a muted browse-row preview, an ad slot that
// never plays — and which one is the real player only becomes apparent once it
// starts playing. Latching one at document_idle picks whichever happened to
// exist and be measurable then, with no way back: a player with no metadata
// yet (nothing has been played in the tab) can't be recognized, and an element
// that never plays reports `paused` forever while the real video runs beside
// it.
//
// Listeners are attached to every <video> seen and are never removed; an event
// is reported only when it comes from the currently resolved element. That
// also suppresses the `pause` a detached element fires when it leaves the
// document, which would otherwise tell the server the video stopped.
(function () {
  // Bail out if already injected (e.g. due to frame navigation).
  if (window.__videoInteractionState) return;

  // Backstop for elements the MutationObserver can't see: it only fires for
  // nodes added after it starts observing, so a <video> already in the DOM at
  // document_idle would otherwise never be considered.
  const RESCAN_MS = 2000;

  // How long to wait for a `seeked` that may never arrive. An element torn
  // down mid-seek never fires one, and a latched isSeeking suppresses every
  // subsequent capture.
  const SEEK_TIMEOUT_MS = 10000;

  const state = (window.__videoInteractionState = {
    videoElement: null,
    isSeeking: false,
    lastSeekMs: 0,   // epoch ms of the most recent seeking/seeked event
    resolve,         // get_video_bounds.js calls this on each capture tick
  });

  const tracked = new WeakSet();

  // Higher is better: actually playing, then merely having loaded metadata,
  // then an element that has never loaded anything. Used only to make the
  // opening guess when there is no current element; once one is chosen it is
  // kept until it leaves the document or another element starts playing.
  function rank(video) {
    if (!video.isConnected) return 0;
    if (!video.paused && !video.ended) return 3;
    if (video.videoWidth > 0 && video.videoHeight > 0) return 2;
    return 1;
  }

  function area(video) {
    const r = video.getBoundingClientRect();
    return r.width * r.height;
  }

  function outranks(candidate, best) {
    const rc = rank(candidate);
    const rb = rank(best);
    if (rc !== rb) return rc > rb;
    return area(candidate) > area(best);
  }

  function setCurrent(video) {
    if (state.videoElement === video) return;
    console.debug('Tracking video interactions on', video);
    state.videoElement = video;
  }

  // Pick the best connected <video>, attaching listeners to any we haven't
  // seen yet. Cheap enough to call on every capture tick and every media
  // event; it's a querySelectorAll plus a rect read per candidate.
  function resolve() {
    const videos = Array.from(document.querySelectorAll('video'));
    videos.forEach(attach);

    // Stay put while the current element is still in the document; takeover is
    // driven by the `play` handler instead. Re-ranking on every read would let
    // a background preview that happens to be playing steal the slot from a
    // player the user has deliberately paused, and report the page as playing.
    const current = state.videoElement;
    if (current && current.isConnected) return current;

    let best = null;
    for (const video of videos) {
      if (rank(video) === 0) continue;
      if (!best || outranks(video, best)) best = video;
    }

    setCurrent(best);
    return best;
  }

  function attach(video) {
    if (tracked.has(video)) return;
    tracked.add(video);

    // Only the current element speaks for the page. Re-resolve first if the
    // stored one has been detached, but don't re-resolve merely because this
    // element paused — that would hand the slot to a preview still playing
    // elsewhere and swallow the pause we're trying to report.
    const isCurrent = () => {
      const current = state.videoElement;
      if (current && current.isConnected) return video === current;
      return video === resolve();
    };

    const report = (isPaused) => {
      browser.runtime.sendMessage({
        type: 'videoStateChange',
        isPaused,
        isSeeking: state.isSeeking,
      });
    };

    video.addEventListener('seeking', () => {
      if (!isCurrent()) return;
      console.debug('Video seeking started');
      state.isSeeking = true;
      state.lastSeekMs = Date.now();
      report(video.paused);
    });

    video.addEventListener('seeked', () => {
      if (!isCurrent()) return;
      console.debug('Video seeking ended');
      state.isSeeking = false;
      state.lastSeekMs = Date.now();
      report(video.paused);
    });

    video.addEventListener('pause', () => {
      if (!isCurrent()) return;
      console.debug('Video paused');
      report(true);
    });

    // Starting to play is how the real player announces itself, and in a tab
    // where nothing has played yet it is the first evidence we have. So an
    // element that starts claims the slot from one that is paused, dead or
    // absent — but not from one that is playing, which is already a better
    // answer than this one.
    video.addEventListener('play', () => {
      const current = state.videoElement;
      const stale =
        !current || !current.isConnected || current.paused || current.ended;
      if (stale) setCurrent(video);
      if (video !== state.videoElement) return;
      console.debug('Video resumed');
      report(false);
    });
  }

  resolve();

  // Watch for videos added dynamically (SPAs, deferred loads, etc.).
  const observer = new MutationObserver(mutations => {
    for (const mut of mutations) {
      for (const node of mut.addedNodes) {
        if (node.nodeName === 'VIDEO' || node.querySelectorAll?.('video')?.length) {
          resolve();
          return;
        }
      }
    }
  });

  observer.observe(document.documentElement, { childList: true, subtree: true });

  setInterval(() => {
    resolve();
    if (state.isSeeking && Date.now() - state.lastSeekMs > SEEK_TIMEOUT_MS) {
      console.debug('Seek timed out without a `seeked` event — clearing');
      state.isSeeking = false;
    }
  }, RESCAN_MS);
})();
