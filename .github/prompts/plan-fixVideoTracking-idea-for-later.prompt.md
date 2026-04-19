# Plan: Fix video tracking + simplify logic

## Root cause

In `track_interactions.js`, `initTracking(getLargestVideo())` is called unconditionally.
If `getLargestVideo()` returns `null` (video not in DOM yet — typical on SPAs like YouTube TV at `document_idle`):
- `initTracking(null)` runs, sets `state.videoElement = null`, then throws `TypeError` on `null.addEventListener(...)`
- The lines that set up the `MutationObserver` are **never reached**
- `window.__videoInteractionState` IS set (with `videoElement: null`), so the early-return guard prevents any retry on re-injection
- Dynamic video elements are never tracked → `videoElement` stays `null` → "No video found — skipping."

## Secondary bugs

- `video.addEventListener('unload', ...)` — `'unload'` is not a video element event; this listener **never fires**. The cleanup `state.videoElement = null` never runs. Fix: use `'emptied'`.
- `checkNewVideoElement` sets `state.videoElement = video` AND then calls `initTracking` which also sets it — redundant.

## Plan

### Phase 1 — Fix `track_interactions.js`

1. Merge `checkNewVideoElement` + `initTracking` into a single `trackVideo(video)` function that:
   - Guards `if (state.videoElement === video) return` (idempotency)
   - Compares areas and bails if the new video is smaller than the tracked one
   - Sets `state.videoElement = video`
   - Attaches `seeking`, `seeked`, `pause`, `play`, `emptied` listeners
2. Replace the unconditional `initTracking(getLargestVideo())` call with:
   `const initial = getLargestVideo(); if (initial) trackVideo(initial);`
3. Keep MutationObserver logic the same (call `trackVideo` instead)

### Phase 2 — Improve clarity

1. Add a short comment block at the top explaining the two phases: initial scan + mutation watch
2. Add inline comments on the two-source seeking logic in `get_video_bounds.js` (browser `video.seeking` vs interaction-state `isSeeking`)

## Files to modify

- `browser_extension/content_scripts/track_interactions.js`
- `browser_extension/content_scripts/get_video_bounds.js` (comment only)

## Verification

1. Load extension on YouTube TV, open a stream — check that initial video is tracked on first capture tick (no "No video found" log)
2. Navigate to a new page (SPA navigation) — verify the new video element is picked up by the observer
3. Pause/unpause — verify state updates reach background.js log
4. Confirm no errors in browser console on pages without video elements
