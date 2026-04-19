# Plan: Frame-Flagging Modal Popup (Part 1)

## TL;DR
Add two new GET endpoints to serve recent-frame metadata and image bytes, a stub `POST /flag_frames` endpoint (returns `{"saved": 0}`) so the Save button is wired end-to-end, and a Bootstrap 5 modal in `is_ad.html` that opens after clicking a report button. Part 2 (phash storage + classify.py pre-check) will replace the stub body.

---

## Decisions
- `POST /flag_frames` is included as a stub in Part 1 (validates input, returns `{"saved": 0}`) so the Save button works end-to-end without 404ing. Part 2 replaces the stub body.
- `recent_frames` deque (maxlen=5) already stores `FrameEntry` with `frame_bytes`, `ext`, `timestamp`, `result`, and `state_classification` — no schema changes needed.
- The `reportWrong()` call to `/report_wrong` is unchanged; `openFlagModal()` fires after it (fire-and-forget, not in `.then()`).
- Bootstrap 5 is already loaded from `/static/bootstrap.bundle.min.js` — use `bootstrap.Modal.getOrCreateInstance`.
- If a frame is evicted before the image request arrives, the endpoint returns 404 — the `<img>` tag simply shows a broken image, which is acceptable.

---

## Phase 1 — New endpoints in `routes/receive.py`

### Step 1: `GET /recent_frames`
- Iterate `list(recent_frames)` and return:
  ```json
  {"frames": [{"timestamp": "...", "classification": "ad|content|null", "state_classification": "..."}]}
  ```
- `classification` = `entry.result.type` if `entry.result` is not None, else `null`.
- No new imports needed beyond what's already in scope.

### Step 2: `GET /recent_frames/{timestamp}/image`
- Path param: `timestamp: str`.
- Find first `entry` in `list(recent_frames)` where `entry.timestamp == timestamp`; raise `HTTPException(404)` if not found.
- Return `Response(content=entry.frame_bytes, media_type="image/jpeg")` if `entry.ext in (".jpg", ".jpeg")`, else `"image/png"`.
- Add `Response` to the `fastapi` import — currently `receive.py` only imports `APIRouter, File, Form, HTTPException, UploadFile`.

### Step 3: `POST /flag_frames` (stub) *(parallel with Steps 1–2)*
- Add Pydantic models to `receive.py`:
  ```python
  class FlagFrameItem(BaseModel):
      timestamp: str
      label: str

  class FlagFramesRequest(BaseModel):
      frames: list[FlagFrameItem]
  ```
- Validate each `label in ("ad", "content")`; raise `HTTPException(400, "invalid label")` if not.
- Return `{"saved": 0}`.
- Add a `# TODO: Part 2 — call add_override()` comment in the body.
- Uses only existing imports (`BaseModel` already imported).

---

## Phase 2 — Modal in `is_ad.html` *(can start in parallel with Phase 1)*

### Step 4: Modal HTML
Add before the closing `</body>` tag (before the `bootstrap.bundle.min.js` `<script>` tag):

```html
<div id="flag-modal" class="modal fade" tabindex="-1" aria-labelledby="flag-modal-label" aria-hidden="true">
  <div class="modal-dialog modal-xl modal-dialog-centered">
    <div class="modal-content bg-dark text-white">
      <div class="modal-header">
        <h5 class="modal-title" id="flag-modal-label">Flag Misclassified Frames</h5>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body">
        <div id="flag-frames-grid" class="row g-3"></div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-primary" id="btn-flag-save">Save</button>
      </div>
    </div>
  </div>
</div>
```

### Step 5: Modal JS *(depends on Step 4)*

Modify `reportWrong(correctLabel, doSwitch)` to call `openFlagModal(correctLabel)` immediately after the fetch (fire-and-forget — not inside `.then()`):

```js
function reportWrong(correctLabel, doSwitch) {
  fetch('/report_wrong', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({correct_label: correctLabel, switch: doSwitch})
  })
  .then(r => r.json())
  .then(data => { if (data.error) alert(data.error); })
  .catch(() => {});
  openFlagModal(correctLabel);
}
```

Add `async function openFlagModal(correctLabel)`:
- Fetch `GET /recent_frames`.
- Clear `#flag-frames-grid`.
- For each frame in the response, insert a `col-sm-6 col-md-4 col-lg-3` Bootstrap column card containing:
  - `<img src="/recent_frames/{timestamp}/image" class="img-fluid rounded mb-2">`
  - A `form-check` div with a checkbox (`class="flag-frame-checkbox"`, `data-timestamp="{timestamp}"`, checked by default) and a label showing `classification` or `"?"` if null.
- Store `correctLabel` on the modal element as `data-correct-label`.
- Show via `bootstrap.Modal.getOrCreateInstance(document.getElementById('flag-modal')).show()`.

Add `async function submitFlaggedFrames()`:
- Read `correctLabel` from `document.getElementById('flag-modal').dataset.correctLabel`.
- Collect `timestamp` from all checked `.flag-frame-checkbox` elements.
- POST to `/flag_frames` with `{frames: [{timestamp, label: correctLabel}, ...]}`.
- On success (200), dismiss via `bootstrap.Modal.getInstance(document.getElementById('flag-modal')).hide()`.
- On error, show `alert(...)`.

Bind `#btn-flag-save` click → `submitFlaggedFrames()` in a **new `<script>` block placed after `bootstrap.bundle.min.js`** so `bootstrap.Modal` is in scope. The existing inline `<script>` block is before the bootstrap script tag, so the new modal JS must live in a separate script block after it.

---

## Relevant files
- `server/src/tv_commercial_detector/routes/receive.py` — 3 new endpoints; `FlagFrameItem` + `FlagFramesRequest` Pydantic models; add `Response` to fastapi import
- `server/src/tv_commercial_detector/templates/is_ad.html` — modal HTML; modified `reportWrong()`; new `openFlagModal()` + `submitFlaggedFrames()` functions; new `<script>` block after bootstrap script tag

---

## Verification
1. `uv run pytest tests/ -m "not integration"` — existing tests pass.
2. Add unit tests in `tests/routes/test_receive.py`:
   - `GET /recent_frames` with empty deque → `{"frames": []}`.
   - `GET /recent_frames` with populated deque → correct `timestamp`/`classification` fields.
   - `GET /recent_frames/{timestamp}/image` with valid timestamp → 200 + correct content-type.
   - `GET /recent_frames/{timestamp}/image` with unknown timestamp → 404.
   - `POST /flag_frames` with valid payload → `{"saved": 0}`.
   - `POST /flag_frames` with invalid label → 400.
3. Manual: start server, open `/is_ad`, click a report button — modal appears with up to 5 thumbnail frames and checked checkboxes.
4. Manual: uncheck some frames, click Save — POST to `/flag_frames` returns 200, modal closes.
5. Manual: click Cancel or the X — modal closes with no side effects.

---

## Further Considerations
1. **JS execution order**: The `submitFlaggedFrames` Save-button binding must live in a `<script>` block *after* `bootstrap.bundle.min.js` so `bootstrap.Modal` is available. The new modal JS should be a separate `<script>` block placed immediately after the bootstrap script tag.
2. **`Response` import**: `receive.py` currently imports `APIRouter, File, Form, HTTPException, UploadFile` from fastapi — `Response` must be added.
3. **Eviction race**: If new frames arrive while the popup is open, the deque may evict older ones. The `/recent_frames/{timestamp}/image` endpoint returns 404 for evicted frames — the image simply won't load in the popup, which is acceptable.
4. **UI on mobile**: The modal grid should stack frames vertically on small screens (Bootstrap's responsive grid handles this via the `col-sm-6 col-md-4 col-lg-3` classes).
