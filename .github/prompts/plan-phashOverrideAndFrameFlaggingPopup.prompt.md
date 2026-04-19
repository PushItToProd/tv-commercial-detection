# Plan: Phash Override + Frame-Flagging Popup

## TL;DR
Two-part feature: (1) after clicking a report button on the `/is_ad` page, a Bootstrap modal appears showing the up-to-5 recent frames, each with a checkbox to include in phash overrides; (2) a new `phash_override.py` module stores those overrides on disk and `classify_image()` checks them first before any other classification pass.

---

## Decisions
- HDMI switch fires immediately on report button click (same as current behavior); popup opens *after* for phash labeling only.
- phash check runs *first* in classify_image, before OpenCV logo/rectangle checks and LLM.
- Popup UI: each frame shows a thumbnail + a checkbox "Include in phash override" (checked by default). The `correctLabel` from the clicked button is the label saved for all checked frames.
- Persistence: `{save_dir}/phash_overrides.json` — JSON array of `{phash, label}`.
- phash distance threshold: configurable via `AppConfig.phash_threshold` (default 10).

---

## Phase 1 — Backend infrastructure

### Step 1: `phash_override.py` (NEW)
File: `server/src/tv_commercial_detector/phash_override.py`

Module-level list `_overrides: list[dict] | None = None` (lazy load).

Functions:
- `_get_overrides_path() -> Path` — `app_config.save_dir / "phash_overrides.json"`
- `get_overrides() -> list[dict]` — load from disk on first call, cache in `_overrides`
- `add_override(image_bytes: bytes, label: str) -> str` — compute `imagehash.phash(Image.open(io.BytesIO(bytes)))`, append `{phash: str(h), label: label}` to `_overrides` list, write full list back to JSON file; return phash string
- `check_override(image_path: str) -> str | None` — if `get_overrides()` is empty return None; else open image via PIL, compute phash, iterate stored overrides comparing hamming distance (`h - imagehash.hex_to_hash(entry["phash"]) <= app_config.phash_threshold`); return matching label or None
- `reset()` — sets `_overrides = None` (for test teardown, consistent with pattern in state.py)

Imports: `imagehash`, `PIL.Image`, `io`, `json`, `pathlib.Path`, `.config.app_config`

### Step 2: `config.py` — add `phash_threshold`
File: `server/src/tv_commercial_detector/config.py`

Add field `phash_threshold: int = 10` to `AppConfig` dataclass. No env-var wiring needed (config.json is sufficient).

### Step 3: `classify.py` — phash pre-check
File: `server/src/tv_commercial_detector/classify.py`

At the top of `classify_image(image_path: str)`, before the `importlib.import_module` dispatch:
```
from .phash_override import check_override
override_label = check_override(image_path)
if override_label is not None:
    return ClassificationResult(source="phash_override", type=override_label, reason="phash_override", reply="(phash override)")
```
Import `check_override` at module level (not inside function).

### Step 4: New routes in `routes/receive.py`
File: `server/src/tv_commercial_detector/routes/receive.py`

Add three new endpoints to the existing `router`:

**a) `GET /recent_frames`**
- Iterate `list(recent_frames)` in order
- Return `{"frames": [{timestamp, classification (= entry.result.type or null), state_classification}]}`

**b) `GET /recent_frames/{timestamp}/image`**  (*depends on 4a*)
- Find `entry` in `recent_frames` where `entry.timestamp == timestamp`; 404 if not found (evicted)
- Return `Response(content=entry.frame_bytes, media_type="image/jpeg" if entry.ext==".jpg" else "image/png")`

**c) `POST /flag_frames`**  (*depends on Step 1*)
- Pydantic models: `FlagFrameItem(timestamp: str, label: str)`, `FlagFramesRequest(frames: list[FlagFrameItem])`
- Validate `label in ("ad", "content")` for each item; 400 on invalid
- For each item, find matching entry in `list(recent_frames)` by timestamp; call `add_override(entry.frame_bytes, item.label)` if found
- Return `{"saved": N}` where N = number of phashes actually stored

Import `add_override` from `..phash_override`.

---

## Phase 2 — Frontend (is_ad.html)

### Step 5: Modal HTML (*parallel with Step 4*)
File: `server/src/tv_commercial_detector/templates/is_ad.html`

Add a Bootstrap modal `<div id="flag-modal" class="modal fade" ...>` before the closing `</body>`:
- Header: "Flag Misclassified Frames"
- Body: `<div id="flag-frames-grid">` — populated dynamically by JS
- Footer: "Save" button (`id="btn-flag-save"`) + "Cancel" dismiss button

Each frame card in the grid:
- `<img>` with `src="/recent_frames/{timestamp}/image"` (loaded lazily when modal opens)
- Checkbox `class="flag-frame-checkbox"` `data-timestamp="{timestamp}"` — checked by default
- Small badge showing the frame's `classification` (ad/content/null)

### Step 6: Modal JS (*depends on Step 5*)
File: `server/src/tv_commercial_detector/templates/is_ad.html`

Modify `reportWrong(correctLabel, doSwitch)`:
```
function reportWrong(correctLabel, doSwitch) {
  // existing fetch to /report_wrong unchanged
  fetch('/report_wrong', {...}).then(...).catch(...);
  // new: open flag popup
  openFlagModal(correctLabel);
}
```

Add functions:
- `async function openFlagModal(correctLabel)` — fetch `/recent_frames`, build grid cards in `#flag-frames-grid`, set `data-correct-label` on modal, show via `bootstrap.Modal.getOrCreateInstance(...).show()`
- `async function submitFlaggedFrames()` — read `correctLabel` from modal's data attribute, collect checked timestamps, POST to `/flag_frames`; dismiss modal on success
- Bind "Save" button click to `submitFlaggedFrames()`

---

## Relevant files
- `server/src/tv_commercial_detector/phash_override.py` — NEW
- `server/src/tv_commercial_detector/classify.py` — add phash pre-check (before importlib dispatch)
- `server/src/tv_commercial_detector/config.py` — add `phash_threshold: int = 10`
- `server/src/tv_commercial_detector/routes/receive.py` — add 3 new endpoints; import add_override from phash_override
- `server/src/tv_commercial_detector/templates/is_ad.html` — modal HTML + JS

---

## Verification
1. `uv run pytest tests/ -m "not integration"` — existing tests still pass
2. Start server (`uv run uvicorn tv_commercial_detector.main:create_app --factory --reload ...`), open `/is_ad`, click a report button — modal appears with recent frame thumbnails and checked checkboxes
3. Check a few frames, click Save — server returns `{"saved": N}`, file `frames/phash_overrides.json` is created with correct entries
4. Send the same (or visually similar) frame to `/receive` — response should show `classification` matching the stored override, not the LLM/OpenCV result
5. Confirm `source: "phash_override"` appears in `classifications.jsonl` for the overridden frame
6. Dismiss popup without saving — no file written, no change to classification behavior

---

## Further Considerations
1. **Cache invalidation**: `_overrides` is a module-level lazy cache. Tests that change `app_config.save_dir` need to call `phash_override.reset()` in their teardown. The existing `reset_state` autouse fixture in conftest.py should be extended to also call `phash_override.reset()`.
2. **Eviction race**: If new frames arrive while the popup is open, the deque may evict older ones. The `/recent_frames/{timestamp}/image` endpoint returns 404 for evicted frames — the image simply won't load in the popup, which is acceptable.
3. **UI on mobile**: The modal grid should stack frames vertically on small screens (Bootstrap's responsive grid handles this).
