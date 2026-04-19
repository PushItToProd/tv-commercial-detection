# Plan: Phash Override Backend

Part two of the plan originally defined in `plan-phashOverrideAndFrameFlaggingPopup.prompt.md`. Part one was implemented based on the sub-plan `plan-frameFlaggingModalPopupPart1.prompt.md`.

## TL;DR
Implement Phase 2 of the phash feature — create `phash_override.py`, wire it into `classify.py` as the first classification step, complete the `/flag_frames` stub, update config, and add tests.

**Current state**: Part 1 is done. `receive.py` has a working `/flag_frames` endpoint that validates labels but returns `{"saved": 0}` with a `# TODO: Part 2` comment. The modal UI posts `{timestamp, label}` pairs (labels: `"ad"`, `"content"`, `"ignore"`). No `phash_override.py` exists yet.

---

## Phase 1 — Core module + config

### Step 1: `phash_override.py` (NEW)
File: `server/src/tv_commercial_detector/phash_override.py`

- Module-level `_overrides: list[dict] | None = None` lazy cache
- `_get_overrides_path() -> Path` — `app_config.save_dir / "phash_overrides.json"`
- `get_overrides() -> list[dict]` — load from disk on first call, cache in `_overrides`; return `[]` if file missing
- `add_override(image_bytes: bytes, label: str) -> str` — compute `imagehash.phash(Image.open(io.BytesIO(image_bytes)))`, append `{"phash": str(h), "label": label}` to cache, write full list to JSON; return phash string
- `check_override(image_path: str) -> str | None` — open image via PIL, compute phash, compare hamming distance against stored overrides (`h - imagehash.hex_to_hash(entry["phash"]) <= app_config.phash_threshold`); return matched label or `None`
- `reset()` — sets `_overrides = None` (test teardown)

Imports: `imagehash`, `PIL.Image`, `io`, `json`, `pathlib.Path`, `.config.app_config`

### Step 2: `config.py` — add `phash_threshold`
File: `server/src/tv_commercial_detector/config.py`

Add `phash_threshold: int = 10` field to `AppConfig` dataclass.

---

## Phase 2 — Wire into classification + route

### Step 3: `classify.py` — phash pre-check
File: `server/src/tv_commercial_detector/classify.py`

Add module-level import `from .phash_override import check_override`. At the very top of `classify_image()`, before the `importlib.import_module` dispatch:

```python
override_label = check_override(image_path)
if override_label is not None:
    return ClassificationResult(source="phash_override", type=override_label, reason="phash_override", reply="(phash override)")
```

### Step 4: Complete `/flag_frames` stub in `receive.py`
File: `server/src/tv_commercial_detector/routes/receive.py`

- Add `from ..phash_override import add_override` import
- Replace the `# TODO: Part 2` comment with actual logic:
  - For each item where `item.label != "ignore"`, find matching entry in `recent_frames` by timestamp
  - Call `add_override(entry.frame_bytes, item.label)` if entry found
  - Track count of successfully saved overrides
- Return `{"saved": N}` with the actual count

---

## Phase 3 — Tests

### Step 5: Extend `conftest.py` `reset_state`
File: `server/tests/conftest.py`

Add `import tv_commercial_detector.phash_override as phash_override_module` and call `phash_override_module.reset()` inside the autouse `reset_state()` fixture (alongside the existing `state_module.recent_frames.clear()` etc.).

### Step 6: New `tests/test_phash_override.py`
Cases to cover:
- `check_override` returns `None` with no overrides on disk
- Round-trip: `add_override(bytes, "ad")` then `check_override(path)` returns `"ad"`
- Hamming distance > threshold → `check_override` returns `None`
- `add_override` writes `phash_overrides.json` to `save_dir`
- Cache reload: `reset()` then `check_override` re-reads from disk
- HTTP test: `POST /flag_frames` with `label: "ignore"` → `{"saved": 0}`, no file written

---

## Relevant files
- `server/src/tv_commercial_detector/phash_override.py` — NEW
- `server/src/tv_commercial_detector/classify.py` — add phash pre-check
- `server/src/tv_commercial_detector/config.py` — add `phash_threshold: int = 10`
- `server/src/tv_commercial_detector/routes/receive.py` — complete `/flag_frames` TODO
- `server/tests/conftest.py` — add `phash_override_module.reset()`
- `server/tests/test_phash_override.py` — NEW

---

## Decisions
- `"ignore"` label: silently skipped in `/flag_frames` — no phash stored, no error
- phash check runs before ALL other classification (OpenCV logo/rectangle, LLM)
- Persistence: `{save_dir}/phash_overrides.json` — JSON array of `{"phash": str, "label": str}`
- Default threshold: 10 hamming distance (configurable via `config.json` as `phash_threshold`)

---

## Verification
1. `uv run pytest tests/ -m "not integration"` — all tests pass
2. Receive frames, open modal, flag some → `/flag_frames` returns `{"saved": N}`, `frames/phash_overrides.json` created with correct entries
3. Send same (or visually similar) frame to `/receive` → classification response has `source: "phash_override"`
4. Flag a frame with `"ignore"` → `{"saved": 0}`, no file written
5. Dismiss modal without saving → no file created, no change to classification behavior
