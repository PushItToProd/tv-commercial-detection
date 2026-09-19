# Code review — 2026-09-18

Scope: `server/src/tv_commercial_detector/`, `browser_extension/`, and
`native_host/audio_capture.py`. Docs, `todo*.md`, `experiments/`, scripts and
templates were not reviewed. The unit suite (342 tests, ~1 s) passes.

## Broken today

### 1. The NHRA profile fails on every frame

`classify.py` calls every profile as `classify_image(image_path, audio_bytes)`,
but `classifiers/nhra_on_fox.py` still declares `classify_image(image_path)`.
The TypeError is caught by the blanket handler in `/receive`, logged as
"Classification error", and the frame becomes `unknown`. Selecting that profile
means the matrix never switches, with no visible failure beyond a log line.
No test covers it; the profile tests only exercise the NASCAR profiles.

### 2. `DETECTOR_ENABLE_DEBOUNCE` is applied as a raw string

The env-var loop in `main.py` does `setattr(app_config, "enable_debounce",
val)` with the string as-is. `"0"` and `"false"` are truthy, so any non-empty
value turns debounce on. It works in the current deployment only because
`example.env` uses `1` and an unset variable arrives as the empty string
(`${RECEIVER_ENABLE_DEBOUNCE:-}` in `docker-compose.yml`).

`tests/test_config.py` copies the loop body instead of calling it, so it can
never catch this. Fix: extract a `load_config()` into `config.py` with
per-field coercion and test that function.

### 3. The extension reports pause/play from any tab as the monitored tab's state

`background.js` handles the `videoStateChange` message without checking
`sender.tab.id` or `captureState.running`. The content script runs on every
page, so pausing a video in an unrelated tab flips the status page to `paused`
until the next capture tick overwrites it. After a stop it is harmless, since
`tabs.get(null)` throws and the POST is skipped. Fix: require
`sender.tab.id === captureState.tabId`.

## Highest-value improvements, in priority order

### 1. Stop blocking the event loop in `/receive`

The handler is `async def` but runs OpenCV, the phash, the LLM round trip and
all disk writes synchronously. While an LLM call runs (up to seconds), the SSE
stream, the status page and every settings POST freeze.

- Server: wrap `classify_image` and `save_frames_batch` in
  `asyncio.to_thread`, then serialize the decision-and-switch section under an
  `asyncio.Lock` so frames commit in arrival order. Without the lock, a
  concurrent server would let frame N+1 switch the matrix before frame N.
- Extension: `doCapture` has no reentrancy guard and `fetch` has no timeout, so
  a slow model stacks ticks. Skip a tick while one is in flight and add
  `AbortSignal.timeout`.

These belong in one change.

### 2. Give a frame one identity on disk

`frame_saver.save_frames_batch` names files `<timestamp>_<i>` where `i` is the
position in the batch, and nothing records that an entry was saved. A
`suspicious_debounce` save dumps all five recent frames; a flip-flop two seconds
later saves the same frames under new names; the periodic saver can save them a
third time. This is the source of the ~40% redundancy `dedupe_frames.py` cleans
up after the fact.

Fix: add `saved_as: str | None` to `FrameEntry`, derive the filename from the
timestamp alone, skip already-saved entries (optionally appending a second
metadata record with the new `save_reason`).

### 3. Load and validate profiles at startup

`classify.py` imports the profile lazily on the first frame, so a missing logo
file, a missing prompt, or the NHRA signature mismatch above all surface as a
stream of `unknown` during a live race. A small registry that imports every
profile in the lifespan and checks the `classify_image` signature (or a
`Protocol`) would have made bug 1 a startup crash. `list_profiles()` globbing
the directory on each call would go away with it.

### 4. Record the OpenCV scores in `signals`

The audio sensor already records `p_audio` on every result whatever decided
the frame. Do the same for `peacock_score`, `usa_score`, `usa_sports_score`
and `side_by_side_score` (and the Fox profile's max template scores). It costs
nothing on the request path and lets thresholds be re-tuned from the archive
without re-running frames. Cheapest high-value change in this list.

### 5. Correlate audio-host requests

`getAudioFromPort` in `background.js` attaches a fresh `onMessage` listener per
request. Two overlapping requests both resolve with the first reply, and a host
that never answers leaves that tick hung forever. Echo an `id` field from the
host and add a timeout.

### 6. Guard `video_offset` like the timebase fields

`receive.py` calls `float(video_offset)` unguarded. Garbage returns a 500.
`NaN` poisons the audio window (every comparison is false, so eviction and the
backward-step reset stop working) and writes bare `NaN` into
`classifications.jsonl`, the exact `jq`-breaking case `video_timebase.py`
exists to prevent. Reuse its `_finite` helper.

## Smaller items

- Each frame is decoded three times per request: `imagehash` (PIL) for the
  override check, `cv2.imread` for the profile, PIL again for the LLM resize.
  Plus a temp file and a `copy2`. Decode once to an array, hash and encode from
  it. The FIXME in `nascar_on_fox.py` already says this.
- `state.classification_reason` is overwritten by every frame's reason even
  when the classification did not change, so the status page's reason often
  describes the latest frame rather than the current verdict. `report_wrong`
  sets `classification` without setting a reason at all.
- The debounce comment in `receive.py` about `unknown` is a real gap: a
  `content, unknown, content` sequence does not switch.
- `/classify` rejects `.jpeg` while `/features` and `/frames` accept it. The
  filename-traversal guard is copy-pasted four times in `review.py`; make it
  one helper.
- `logo_match.MASKED_NETWORK_LOGOS` and `MASKED_SIDE_BY_SIDE_LOGOS` load six
  templates at import that no profile uses. `rectangle_match.KNOWN_RECTANGLES`
  is Fox-specific in a generic module (the TODO there already says so).
- Eight `print()` calls in `receive.py` bypass the logger.
- `set_classifier_profile` in `status.py` has a commented-out validator and an
  inline `from fastapi import HTTPException`.
- No auth on any route, bound to `0.0.0.0`, and the extension requests
  `<all_urls>` plus `downloads` (unused). Acceptable on a home LAN; noting it.

## Worth preserving

- **`classifiers/nascar_on_nbc.py`** is the best code in the repo. Every
  threshold has a measurement and a stated failure asymmetry; the mask-fraction
  guard around the translucent bugs is the kind of check that only comes from
  having been burned; the ordering rationale (side-by-side, then bug, then
  audio, then LLM) is written where the next editor will see it. Hold every
  future profile to that standard.
- **The audio sensor's abstention model**: named reasons, the identity check
  that stops a re-classify from borrowing the live window, a window kept in
  broadcast time, and `signals` recorded on every result. Extend rather than
  replace (see improvement 4).
- **`VideoStatus`** with severity ordering and client-side aging, the
  `StopReason` enum, and the Infinity/NaN handling in `video_timebase.py`.
- **The extension's player resolution** in `track_interactions.js`: only
  `play` transfers ownership, events from non-current elements are ignored,
  sticky seek state times out. A hard problem solved cleanly.
- **The native host's sink following** and its uncorked check before warning
  about silence.
- **The test suite**: fast, with an autouse reset fixture that keeps the
  module-level singletons honest. Its one blind spot is testing copies of logic
  instead of the real entry points (see bug 2).
