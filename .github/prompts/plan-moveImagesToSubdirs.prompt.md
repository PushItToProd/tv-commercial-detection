# Plan: Move Images to images/ and thumbnails/ Subdirectories

## TL;DR
Currently all images (full-size and thumbnails) live flat in `save_dir/`. Change the code to save full-size frames to `save_dir/images/` and on-demand thumbnails to `save_dir/thumbnails/` (replacing the `compressed_` prefix convention). Update all path constructions, route logic, tests, and scripts accordingly.

## Steps

### Phase 1 — Core app changes

1. **frame_saver.py**:
   - Add `images_dir = save_dir / "images"` and `images_dir.mkdir(parents=True, exist_ok=True)` alongside existing `save_dir.mkdir()`
   - Change `dest = save_dir / filename` → `dest = images_dir / filename`
   - `classifications.jsonl` stays at `save_dir / "classifications.jsonl"` (unchanged)

2. **routes/review.py — `/save` POST (line 91)**:
   - Change `save_path = save_dir / filename` → `save_path = save_dir / "images" / filename`
   - Add `(save_dir / "images").mkdir(parents=True, exist_ok=True)` before saving

3. **routes/review.py — `serve_frame` GET `/frames/{filename}` (lines 98–125)**:
   - Remove the `if filename.startswith("compressed_"):` guard (lines 109–110) — no longer needed
   - Change `original_path = save_dir / filename` → `original_path = save_dir / "images" / filename`
   - Change `compressed_path = save_dir / f"compressed_{filename}"` → `compressed_path = save_dir / "thumbnails" / filename`
   - Add `(save_dir / "thumbnails").mkdir(parents=True, exist_ok=True)` before generating thumbnail

4. **routes/review.py — `serve_frame_full` GET `/frames/full/{filename}` (lines 128–141)**:
   - Remove the `if filename.startswith("compressed_"):` guard (lines 135–136) — no longer needed
   - Change `path = app_config.save_dir / filename` → `path = app_config.save_dir / "images" / filename`

5. **routes/review.py — `/review` GET (lines 211–230)**:
   - Change glob source: `save_dir.glob("*.png")` → `(save_dir / "images").glob("*.png")` (same for `.jpg`)
   - Remove `if not p.name.startswith("compressed_")` filter — no longer needed (thumbnails are in separate dir)

### Phase 2 — Test updates

6. **tests/routes/test_review.py**:
   - In `_save_jpeg()` helper (line 22): change path from `app_config.save_dir / filename` → `app_config.save_dir / "images" / filename`, add `path.parent.mkdir(parents=True, exist_ok=True)` before write
   - In `test_save_stores_image` (line 118): change path check from `app_config.save_dir / saved` → `app_config.save_dir / "images" / saved`

### Phase 3 — Script updates

7. **scripts/find_dupes.py**:
   - Update `IMAGE_DIR` constant (line 9) from `Path(__file__).parent.parent / "frames"` → `Path(__file__).parent.parent / "frames" / "images"`
   - Update `find_duplicates()` to not filter `compressed_*` prefix since those no longer exist in the images dir (remove `.startswith('compressed_')` check on line 70)

8. **scripts/view_classification_results.py**:
   - Update default `frames_dir` (line 444) from `jsonl_path.parent / "frames"` → `jsonl_path.parent / "frames" / "images"` so it defaults to the new images subdirectory

## Relevant Files
- `server/src/tv_commercial_detector/frame_saver.py` — lines 20–31 (save_dir setup and dest path)
- `server/src/tv_commercial_detector/routes/review.py` — lines 87–141 (save/serve endpoints), lines 213–219 (review glob)
- `server/tests/routes/test_review.py` — lines 21–24 (`_save_jpeg`), lines 114–121 (`test_save_stores_image`)
- `server/scripts/find_dupes.py` — lines 9, 70 (IMAGE_DIR constant, compressed_ filter)
- `server/scripts/view_classification_results.py` — line 444 (frames_dir default)

## Verification
1. Run `uv run pytest tests/ -m "not integration"` — all tests pass
2. Run `uv run ruff check src/` — no lint errors
3. Manual: POST a frame to `/receive`, confirm it's saved in `frames/images/`, then GET `/frames/{filename}` and confirm thumbnail appears in `frames/thumbnails/`
4. Manual: GET `/review` — images load correctly from the new paths

## Decisions
- Metadata files (`classifications.jsonl`, `labels.json`, `features.jsonl`) remain at `save_dir/` root — not images, so no change
- The `compressed_` prefix convention is fully replaced by the `thumbnails/` subdirectory; the guards that blocked serving `compressed_*` files are removed as they're now irrelevant
- Existing files on disk are NOT migrated; this only affects new saves going forward. User can manually `mv frames/*.jpg frames/*.png frames/images/` to migrate
- `view_classification_results.py` default updated; users with pre-migration data can pass `--frames-dir frames/` explicitly
