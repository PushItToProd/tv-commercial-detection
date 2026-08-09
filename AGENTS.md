# AGENTS.md

## Project overview

**TV Commercial Detector** — a system that automatically detects TV commercials during live race broadcasts on YouTube TV and switches an HDMI matrix to a different input during ad breaks.

- **`browser_extension/`** — Firefox extension (Manifest V2) that periodically captures screenshots (and optionally audio) of the active video tab and sends them to the server.
- **`native_host/`** — Firefox native messaging host that captures system audio from a PulseAudio/PipeWire monitor source and streams it to the browser extension on demand.
- **`server/`** — FastAPI app that classifies each screenshot using OpenCV and a local multimodal LLM (llama.cpp), tracks the current broadcast state, and controls an HDMI matrix switcher over HTTP.

External services (run via Docker):
- **llama.cpp** — local LLM server used for vision-based classification (`LLAMA_SERVER_URL`).
- **hdmi-matrix-control** — HTTP service for switching HDMI inputs (`DETECTOR_MATRIX_URL`).

---

## Repository layout

```
browser_extension/   Firefox extension source (Manifest V2)
native_host/         Firefox native messaging host for audio capture
  audio_capture.py     Main script: rolling PCM buffer + native messaging protocol
  com.tvdetector.audio_capture.json  Manifest template (path filled in by install.sh)
  install.sh           One-time setup: writes manifest to ~/.mozilla/native-messaging-hosts/
  run                  Wrapper that activates the venv and execs audio_capture.py
  venv/                Python virtual environment (created by install.sh)
server/              FastAPI application
  src/tv_commercial_detector/
    classify.py        Entry point for classification; dispatches to active classifier profile
    config.py          App configuration dataclass (AppConfig)
    frame_saver.py     Periodic frame saving; rolling in-memory buffer of recent frames
    main.py            App factory and lifespan startup
    matrix.py          HDMI matrix control helpers
    metrics.py         Prometheus metrics setup
    phash_override.py  Perceptual hash override system (manual label corrections)
    state.py           In-memory application state (AppState)
    classification/    Low-level classification primitives
      llm_match.py     OpenAI-compat LLM calls (image resize, prompt, response parsing)
      logo_match.py    OpenCV template matching for network/side-by-side logos
      rectangle_match.py  OpenCV contour detection for known ad-break bounding boxes
      result.py        ClassificationResult dataclass
    classifiers/       Pluggable classifier profiles (selected via classifier_profile config)
      nascar_on_fox.py Multi-pass classifier: logo → rectangle → LLM quick check → LLM prompt
      nhra_on_fox.py   Variant for NHRA drag racing broadcasts on Fox/FS1
      nascar_on_hbo_max.py  WIP profile for TNT Sports coverage on HBO Max; OpenCV logo checks only, no LLM pass yet
      nascar_on_nbc.py  Cup coverage on NBC; colour-matched peacock bug, then LLM fallback
    routes/            FastAPI routers
    prompt/            LLM prompt text and logo images used for OpenCV matching
    static/            Static assets (Bootstrap CSS/JS for UI templates)
    templates/         Jinja2 templates (review UI, is_ad page)
  tests/               Unit and integration tests
    classification/    Tests for logo_match and rectangle_match
    routes/            Tests for each route
    integration/       Integration tests (require a live llama.cpp server)
  scripts/             Utility scripts (find_dupes.py, view_classification_results.py, etc.)
  config.json          Optional local config (gitignored; overrides defaults)
  frames/              Save dir (runtime, gitignored)
    images/              Full-size frames
    thumbnails/          Review-UI thumbnails, generated on demand
    audio/               Audio clips, same stem as their frame
    labels.json          Manual labels, keyed by frame filename
    features.jsonl       Manual feature annotations
    classifications.jsonl  Metadata written by the frame saver
    phash_overrides.json   Perceptual-hash label overrides
docker-compose.yml   Orchestrates llama, hdmi-matrix-control, and receiver containers
example.env          Template — copy to .env and fill in values before running Docker
```

---

## Server — Python / FastAPI

### Package manager

Use **`uv`** for all Python operations. Never use `pip` directly.

```bash
# Install dependencies
cd server
uv sync

# Run the dev server
uv run uvicorn tv_commercial_detector.main:create_app --factory --reload --host 0.0.0.0 --port 11679

# Run a script / one-off command inside the venv
uv run python -m tv_commercial_detector.classify

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Run tests (unit only; integration tests require a live llama.cpp server)
uv run pytest tests/ -m "not integration"
```

### Python version

Requires Python ≥ 3.14 (see `server/pyproject.toml`).

### Key dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | Web framework and ASGI server |
| `openai` | OpenAI-compatible client for llama.cpp |
| `pillow` | Image resizing before sending to LLM |
| `opencv-python-headless` | Template matching and contour detection |
| `imagehash` | Perceptual image hashing |
| `jinja2` | HTML templates for review UI and status page |
| `prometheus-fastapi-instrumentator` | Metrics endpoint (`/metrics`) |
| `aiofiles` | Async file I/O for saving/reading frames |

### Configuration

Config is layered (later overrides earlier):
1. `server/config.json` (optional; path overridden by `CONFIG_FILE` env var)
2. Environment variables:
   - `DETECTOR_MATRIX_URL` — HDMI matrix URL
   - `DETECTOR_SAVE_DIR` — save dir root (frames go in its `images/` subdir)
   - `DETECTOR_ENABLE_DEBOUNCE` — enable debounce logic
   - `DETECTOR_CLASSIFIER_PROFILE` — which classifier profile to use (default: `nascar_on_fox`)
   - `LLAMA_SERVER_URL` — URL for the llama.cpp server (default: `http://localhost:3002`)
3. `PROMPT_FILE` — path to the classification prompt (default: `server/src/tv_commercial_detector/prompt/prompt.txt`)

The `config.json` also supports an `output_settings` map that defines which HDMI matrix input/output to activate per classification (`ad` or `content`).

Media is split into subdirectories of `save_dir` — `images/`, `thumbnails/`,
`audio/` — so that listing frames doesn't have to walk the thumbnails and audio
too. Use the `images_dir()` / `thumbnails_dir()` / `audio_dir()` helpers in
`config.py` rather than joining the paths by hand; they resolve `save_dir` on
each call, which matters because it's assigned during startup. Metadata files
stay at the `save_dir` root, and the records inside them key on bare filenames.

A save dir predating this layout (frames loose at the root, thumbnails under a
`compressed_` prefix) needs a one-time migration:

```bash
uv run python scripts/migrate_frames_to_subdirs.py           # dry run
uv run python scripts/migrate_frames_to_subdirs.py --apply
```

Bursts of saved frames and repeated commercials leave the save dir roughly 40%
redundant. `scripts/dedupe_frames.py` groups frames by perceptual hash and keeps
one representative of each group, pruning the metadata files to match. It never
removes a frame carrying a label or feature record, and leaves `audio/` alone
unless `--include-audio` is passed, since identical images can carry different
commentary.

```bash
uv run python scripts/dedupe_frames.py                     # dry run
uv run python scripts/dedupe_frames.py --report-conflicts  # audit labels only
uv run python scripts/dedupe_frames.py --apply --include-audio --drop-blank
```

Thresholds above 11 start merging frames with differing manual labels; see
`notes/frame-deduplication.md` for the measurements behind the default of 10 and
for why grouping is deliberately non-transitive.

Additional `AppConfig` fields:
- `phash_threshold` — max perceptual hash distance for override matches (default: `10`)
- `enable_llm_audio` — enable audio-based LLM classification (default: `false`)
- `llm_model_name` — model name sent to llama.cpp, in case we're using llama.cpp in router mode (default: `LLAMA_MODEL_NAME` env var, or `"local"`)

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/receive` | Accept a screenshot + playback state from the extension |
| `POST` | `/video-state` | Update playback state only (no image) |
| `POST` | `/report_wrong` | Report that the current classification is wrong |
| `POST` | `/capture` | Save current in-memory recent frames to disk |
| `GET` | `/recent_frames` | List in-memory recent frames with timestamps and classifications |
| `GET` | `/recent_frames/{timestamp}/image` | Retrieve a recent in-memory frame by timestamp |
| `POST` | `/flag_frames` | Label recent frames and store phash overrides |
| `GET` | `/review` | Manual review UI for saved frames (paginated + filterable; see below) |
| `POST` | `/save` | Save a frame to disk |
| `GET` | `/frames/{filename}` | Retrieve a saved (thumbnail) frame |
| `GET` | `/frames/full/{filename}` | Retrieve a full-size saved frame |
| `POST` | `/classify` | Re-classify a saved frame on demand |
| `POST` | `/features` | Extract and return OpenCV features from a frame |
| `GET` | `/is_ad` | HTML status page (used on secondary devices) |
| `GET` | `/is_ad/status` | JSON status snapshot |
| `GET` | `/is_ad/stream` | SSE stream of current state (classification, paused, seeking) |
| `GET` | `/is_ad/last_frame` | Most recently received frame image |
| `POST` | `/trigger_matrix` | Manually trigger an HDMI matrix switch |
| `POST` | `/settings/auto_switch` | Enable/disable auto-switch |
| `POST` | `/settings/enable_debounce` | Enable/disable debounce |
| `GET/POST` | `/settings/classifier_profile` | Get or set the active classifier profile |
| `POST` | `/settings/pause_auto_switch` | Temporarily pause auto-switch |
| `POST` | `/settings/resume_auto_switch` | Clear temporary auto-switch pause and re-apply |

#### `/review` query params

The save dir holds tens of thousands of frames, so `/review` filters and pages
server-side and only embeds the current page in the HTML. Every option is a
query param, so any view is linkable and bookmarkable.

| Param | Values | Description |
|---|---|---|
| `page` | ≥ 1 (default `1`) | 1-based page number; clamped to the last page |
| `per_page` | 1–500 (default `100`) | Frames per page |
| `sort` | `asc` (default) / `desc` | Chronological order — oldest or newest first |
| `label` | `ad` / `content` / `ignore` / `__unset__` | Stored label |
| `network_logo` | any `VALID_NETWORK_LOGOS` value / `__unset__` | Stored feature |
| `logo_position` | any `VALID_LOGO_POSITIONS` value / `__unset__` | Stored feature |
| `scoreboard_position` | any `VALID_SCOREBOARD_POSITIONS` value / `__unset__` | Stored feature |
| `q` | substring | Case-insensitive filename match |
| `start` / `end` | `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, or `YYYY-MM-DDTHH:MM:SS` | Inclusive bounds on capture time (a space may replace the `T`) |
| `incomplete` | bool | Only frames missing a label or any feature |
| `step` | index or `last` | Opens the step view at that offset within the page |

`__unset__` is the "no value recorded" sentinel; it can't be spelled `none`
because both `network_logo` and `scoreboard_position` accept a literal `none`.
An unrecognized filter value returns 400, and an out-of-range `page`/`per_page`
returns 422, so a bad bookmark fails loudly rather than silently showing
everything.

The step view walks the current page and rolls onto the neighbouring page at
either end. "Prev/next incomplete" only scans the current page — use
`incomplete=1` to sweep the whole set.

Two naming conventions are in play: `2026-03-11_15-34-05.jpg` from `/save` and
`2026-03-11T20-24-30-765665_0.jpg` from the frame saver. `frame_timestamp()`
parses either into a comparable tuple, and both ordering and the `start`/`end`
bounds go through it:

- `frame_sort_key()` wraps it for sorting. A raw filename sort puts every `_`
  file after every `T` file from the same date (`_` > `T` in ASCII) and orders
  batch suffixes as text, so `_10` lands before `_9`. Names that don't parse
  sort last.
- `parse_time_bound()` turns a `start`/`end` value into the same shape.
  Components the bound leaves out widen it: `end=2026-01-01` runs through
  23:59:59 that day and `end=2026-01-01T14:30` through the end of that minute,
  so a bare date still means the whole day. Frames whose names carry no
  timestamp are excluded whenever a bound is set, since they can't be placed in
  time. A malformed bound is a 400.

### Running with Docker

Running under Docker requires a .env file to be created first.

```bash
docker compose up -d
```

The receiver container is exposed on `RECEIVER_PORT` (default `11679`).

---

## Browser extension

- Target browser: **Firefox** (Manifest V2).
- Load for development: `about:debugging#/runtime/this-firefox` → "Load Temporary Add-on…" → select any file inside `browser_extension/`.
- **No build step** — the extension runs directly from source.
- Key files:
  - `background.js` — alarm-driven screenshot loop, sends frames to the server via `multipart/form-data` POST to `/receive`.
  - `content_scripts/track_interactions.js` — injected into every page; reports play/pause/seek events back to the background script.
  - `content_scripts/get_video_bounds.js` — injected on demand; finds the `<video>` element bounds for cropping.
  - `popup.html` / `popup.js` — configuration UI (server URL, capture interval, start/stop).

Configuration (server endpoint URL, capture interval) is stored via `browser.storage.local`.

---

## Native host

`native_host/audio_capture.py` is a Firefox [native messaging](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Native_messaging) host that gives the browser extension access to system audio.

- **Protocol** — Firefox launches the host process and communicates via stdin/stdout using 4-byte little-endian length-prefixed JSON messages.
- **Audio capture** — Opens a `sounddevice.InputStream` against the PulseAudio/PipeWire monitor source for the default sink (auto-detected via `pactl`). Incoming PCM is stored in a thread-safe rolling deque capped at `AUDIO_BUFFER_SECONDS` (default: 10 s).
- **Commands** accepted from the extension:
  - `get_audio` — returns the last `duration_ms` milliseconds of audio as a base64-encoded WAV in `{"audio": "..."}`.
  - `ping` — responds with `{"pong": true}` (health check).
- **Standalone mode** — pass `--save-dir DIR` to periodically write WAV snapshots to disk for testing without the extension.

### Setup

```bash
cd native_host
# Create the venv and install dependencies (sounddevice, numpy), then register the manifest
./install.sh

# Reload the extension in Firefox (about:debugging) after installing
```

`install.sh` writes a resolved copy of `com.tvdetector.audio_capture.json` (with the absolute path to `run`) into `~/.mozilla/native-messaging-hosts/`. Re-run it if the repo is moved.

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `AUDIO_BUFFER_SECONDS` | `10` | Rolling buffer length in seconds |
| `AUDIO_SAMPLE_RATE` | `44100` | Sample rate in Hz |
| `AUDIO_CHANNELS` | `1` | Number of channels |
| `AUDIO_DEVICE` | system default | `sounddevice` input device name or index |

Logs are written to `native_host/audio_capture.log` (stderr redirected by the `run` wrapper).

---

## Classification

`classify.py` dispatches to the active classifier profile (set via `classifier_profile` config, default `nascar_on_fox`). Before invoking a profile, it checks `phash_override.py` for a stored perceptual-hash match and short-circuits with the stored label if found.

The `nascar_on_fox` profile uses a multi-pass pipeline:

0. **Phash override** (`phash_override.py`) — if the frame matches a stored phash entry (within `phash_threshold`), return the stored label immediately.
1. **Network logo match** (OpenCV) — if a Fox/FS1 logo is found in the upper right, classify as `content`.
2. **Side-by-side logo match** (OpenCV) — if a side-by-side ad-break logo is found in the upper left (Fox, FS1, Truck Series, or Amazon Prime NASCAR Nonstop), classify as `ad`.
3. **Rectangle detection** (OpenCV) — if a known ad-break bounding box pattern is detected, classify as `ad`.
4. **LLM quick check** — ask the LLM whether the frame contains any NASCAR-related content; if not, classify as `ad`.
5. **LLM full prompt** — send the image and prompt to llama.cpp for a final classification decision.

The `nhra_on_fox` profile follows the same structure but uses NHRA-specific logo assets.

The `nascar_on_hbo_max` profile is a work in progress for TNT Sports coverage on HBO Max. It only runs two OpenCV logo checks — a full-screen "we'll be back" card (`ad`) and a side-by-side "commercial break in progress" overlay (`content`, since racing is still shown side-by-side) — and falls back to `content` by default. It has no rectangle-detection or LLM pass yet.

The `nascar_on_nbc` profile covers Cup coverage on NBC. It checks the "NASCAR NON STOP" side-by-side banner in the upper left (`ad`), then the NBC peacock bug in the upper right (`content`), then falls through to the LLM quick check and `prompt_nbc.txt`. Two details are load-bearing:

- The peacock is matched **in colour**. `load_masked` / `mask_non_white` zero out everything that isn't near-white and would erase a six-colour logo entirely, so this template is loaded with a plain `cv2.imread`.
- Its search window is tight (`x 1740–1880, y 40–140` at 1920×1080). Over the wide upper-right region the Fox profile uses, the weakest true positive scores below the strongest false positive and the match is unusable.

The peacock threshold (`0.55`) was measured against the labelled NBC frames and 3000 random non-NBC frames. The side-by-side half is **unvalidated** — no NBC ad-break frame exists in the dataset — and is guarded by `ENABLE_SIDE_BY_SIDE_CHECK`.

Images are resized to at most 800 px on the longest side and JPEG-encoded (quality 50) before being sent to the LLM. The default prompt lives in `server/prompt/prompt.txt`; profiles can supply their own by passing `prompt=` to `llm_match.classify_by_prompt`.

Classification labels: `ad`, `content` (racing), `unknown`.

---

## Development notes

- State is kept in a module-level `state` object in `state.py` — not persisted between restarts.
- A test suite exists under `server/tests/`. Run unit tests with `uv run pytest tests/ -m "not integration"`. Integration tests (in `tests/integration/`) require a live llama.cpp server.
- Type checking: the project uses Pyright (see inline `# pyright: ignore` comments).
- Linting/formatting: configured via `ruff` in `pyproject.toml`. Run `uv run ruff check src/` and `uv run ruff format src/`.
