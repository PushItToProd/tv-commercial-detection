# Plan: Classification Profile System

## TL;DR
Introduce a `classifiers/` sub-package where each `.py` file exports a `classify_image(image_path: str) -> ClassificationResult` function. Move the current NASCAR-on-Fox 3-pass logic from `classify.py` into `classifiers/nascar_on_fox.py`. Turn `classify.py` into a thin dispatcher that dynamically imports the active profile via `importlib`. Wire up a `classifier_profile` config field, a `/settings/classifier_profile` GET+POST endpoint, and a profile selector `<select>` in `is_ad.html`.

---

## Phase 1 – classifiers/ package

1. Create `server/src/tv_commercial_detector/classifiers/__init__.py` (empty)
2. Create `server/src/tv_commercial_detector/classifiers/nascar_on_fox.py`
   - Move the 3-pass `classify_image` logic from `classify.py` verbatim (logo_match → side_by_side → rectangle_match → llm_quick_reject → llm_prompt)
   - Imports: `..classification.{llm_match,logo_match,rectangle_match}` and `..classification.result.ClassificationResult`
3. Rewrite `classify.py` as a thin dispatcher:
   - `classify_image(image_path: str) -> ClassificationResult`: reads `app_config.classifier_profile`, calls `importlib.import_module`.
   - `list_profiles() -> list[str]`: enumerates `classifiers/*.py` excluding `__init__.py` via `Path(__file__).parent / "classifiers"`, returns sorted list of stem names.

## Phase 2 – Config and startup

4. Add `classifier_profile: str = "nascar_on_fox"` to `AppConfig` in `config.py`.
5. Add `"DETECTOR_CLASSIFIER_PROFILE": "classifier_profile"` to the `env_map` in `main.py` lifespan (the existing `config.json` loop already handles it via `hasattr`).

## Phase 3 – Settings endpoints

6. Add to `routes/status.py`:
   - GET `/settings/classifier_profile` → returns `{"current": "...", "available": [...]}`
   - POST `/settings/classifier_profile` → accepts `{"profile": "..."}` body; validate profile name with regex `^[a-z][a-z0-9_]*$` and confirm the module exists (use `list_profiles()`), then update `app_config.classifier_profile` and call `broadcast_status()`.

## Phase 4 – is_ad.html UI

7. Add `<select id="classifier-profile-select">` with a label in the `#bottom-controls` bar.
8. On page load: fetch GET `/settings/classifier_profile`, populate options, set current value.
9. On change: POST to `/settings/classifier_profile` with `{profile: selectEl.value}`.

---

## Relevant files

- `server/src/tv_commercial_detector/classify.py` — becomes dispatcher; `list_profiles()` added
- `server/src/tv_commercial_detector/classifiers/__init__.py` — new, empty
- `server/src/tv_commercial_detector/classifiers/nascar_on_fox.py` — new, migrated logic
- `server/src/tv_commercial_detector/config.py` — add `classifier_profile` field
- `server/src/tv_commercial_detector/main.py` — add env var mapping
- `server/src/tv_commercial_detector/routes/status.py` — add GET+POST endpoints
- `server/src/tv_commercial_detector/templates/is_ad.html` — add profile selector

No changes to `classification/` subpackage, `pyproject.toml`, `state.py`, or `receive.py`.

---

## Verification

Implement using automated tests.

1. `uv run uvicorn tv_commercial_detector.main:create_app --factory` starts without errors
2. GET `/settings/classifier_profile` returns `{"current": "nascar_on_fox", "available": ["nascar_on_fox"]}`
3. POST `/settings/classifier_profile` with `{"profile": "nascar_on_fox"}` returns 200
4. POST `/settings/classifier_profile` with invalid name returns 422/400
5. `uv run python -m tv_commercial_detector.classify path/to/image.jpg` still works
6. Dropdown appears in `/is_ad` UI and reflects current profile
7. `DETECTOR_CLASSIFIER_PROFILE=nascar_on_fox` env var sets the profile on startup

---

## Decisions

- Profile names are module stems (e.g. `"nascar_on_fox"`), not human-readable labels. Human labels can be a future addition.
- `classify.py` stays as the public entry point; `receive.py` is unchanged.
- No second classifier is created — user adds their own `.py` files to `classifiers/`.
- Profile name validated with `^[a-z][a-z0-9_]*$` before import to prevent path traversal.
- `importlib.import_module` caches in `sys.modules`; switching profiles loads the new module fresh (fine for runtime changes).
- Classifier profile is NOT added to the SSE stream status (it's config, not real-time state).
- `pyproject.toml` needs no change — Python source files in `classifiers/` are auto-discovered by setuptools `find_packages`.
