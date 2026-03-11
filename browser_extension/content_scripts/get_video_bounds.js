// Find the largest visible video element on the page and return its bounding rect
(function () {
  const clamp0 = n => Math.max(0, n);

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

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  return {
    x:      Math.round(clamp0(rect.left   * dpr)),
    y:      Math.round(clamp0(rect.top    * dpr)),
    width:  Math.round(clamp0(rect.width  * dpr)),
    height: Math.round(clamp0(rect.height * dpr)),
  };
})();
