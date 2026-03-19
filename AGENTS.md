# AGENTS.md

## Project overview

**TV Commercial Detector** — a two-component system that automatically detects TV commercials during live race broadcasts on YouTube TV and switches an HDMI matrix to a different input during ad breaks.

- **`browser_extension/`** — Firefox extension (Manifest V2) that periodically captures screenshots of the active video tab and sends them to the server.
- **`server/`** — FastAPI app that classifies each screenshot using OpenCV and a local multimodal LLM (llama.cpp), tracks the current broadcast state, and controls an HDMI matrix switcher over HTTP.

External services (run via Docker):
- **llama.cpp** — local LLM server used for vision-based classification (`LLAMA_SERVER_URL`).
- **hdmi-matrix-control** — HTTP service for switching HDMI inputs (`DETECTOR_MATRIX_URL`).

---

## Repository layout

```
browser_extension/   Firefox extension source (Manifest V2)
server/              FastAPI application
  src/tv_commercial_detector/
    classify.py        Entry point for classification; dispatches to active classifier profile
    config.py          App configuration dataclass (AppConfig)
    frame_saver.py     Periodic frame saving; rolling in-memory buffer of recent frames
    main.py            App factory and lifespan startup
    matrix.py          HDMI matrix control helpers
    metrics.py         Prometheus metrics setup
    state.py           In-memory application state (AppState)
    classification/    Low-level classification primitives
      llm_match.py     OpenAI-compat LLM calls (image resize, prompt, response parsing)
      logo_match.py    OpenCV template matching for network/side-by-side logos
      rectangle_match.py  OpenCV contour detection for known ad-break bounding boxes
      result.py        ClassificationResult dataclass
    classifiers/       Pluggable classifier profiles (selected via classifier_profile config)
      nascar_on_fox.py Three-pass classifier: logo → rectangle → LLM quick check → LLM prompt
    routes/            FastAPI routers
    prompt/            LLM prompt text and logo images used for OpenCV matching
    templates/         Jinja2 templates (review UI, is_ad page)
  tests/               Unit and integration tests
    classification/    Tests for logo_match and rectangle_match
    routes/            Tests for each route
    integration/       Integration tests (require a live llama.cpp server)
  scripts/             Utility scripts (find_dupes.py, view_classification_results.py, etc.)
  config.json          Optional local config (gitignored; overrides defaults)
  frames/              Saved frames and labels (runtime, gitignored)
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
uv run uvicorn tv_commercial_detector.main:create_app --factory --reload --host 0.0.0.0 --port 11434

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

### Configuration

Config is layered (later overrides earlier):
1. `server/config.json` (optional; path overridden by `CONFIG_FILE` env var)
2. Environment variables:
   - `DETECTOR_MATRIX_URL` — HDMI matrix URL
   - `DETECTOR_SAVE_DIR` — directory for saved frames
   - `DETECTOR_ENABLE_DEBOUNCE` — enable debounce logic
   - `DETECTOR_CLASSIFIER_PROFILE` — which classifier profile to use (default: `nascar_on_fox`)
3. `LLAMA_SERVER_URL` — URL for the llama.cpp server (default: `http://192.168.1.27:3002`)
4. `PROMPT_FILE` — path to the classification prompt (default: `server/prompt/prompt.txt`)

The `config.json` also supports an `output_settings` map that defines which HDMI matrix input/output to activate per classification (`ad` or `content`).

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/receive` | Accept a screenshot + playback state from the extension |
| `POST` | `/video-state` | Update playback state only (no image) |
| `POST` | `/report_wrong` | Report that the current classification is wrong |
| `GET` | `/review` | Manual review UI for saved frames |
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
| `POST` | `/settings/resume_auto_switch` | Clear temporary auto-switch pause and re-apply |

### Running with Docker

Running under Docker requires a .env file to be created first.

```bash
docker compose up -d
```

The receiver container is exposed on `RECEIVER_PORT` (default `11434`).

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

## Classification

`classify.py` dispatches to the active classifier profile (set via `classifier_profile` config, default `nascar_on_fox`). The `nascar_on_fox` profile uses a multi-pass pipeline:

1. **Network logo match** (OpenCV) — if a Fox/FS1/CW Sports logo is found in the upper right, classify as `content`.
2. **Side-by-side logo match** (OpenCV) — if a side-by-side ad-break logo is found in the upper left, classify as `ad`.
3. **Rectangle detection** (OpenCV) — if a known ad-break bounding box pattern is detected, classify as `ad`.
4. **LLM quick check** — ask the LLM whether the frame contains any NASCAR-related content; if not, classify as `ad`.
5. **LLM full prompt** — send the image and prompt to llama.cpp for a final classification decision.

Images are resized to at most 800 px on the longest side and JPEG-encoded (quality 50) before being sent to the LLM. The prompt lives in `server/prompt/prompt.txt`.

Classification labels: `ad`, `content` (racing), `unknown`.

---

## Development notes

- State is kept in a module-level `state` object in `state.py` — not persisted between restarts.
- A test suite exists under `server/tests/`. Run unit tests with `uv run pytest tests/ -m "not integration"`. Integration tests (in `tests/integration/`) require a live llama.cpp server.
- Type checking: the project uses Pyright (see inline `# pyright: ignore` comments).
- Linting/formatting: configured via `ruff` in `pyproject.toml`. Run `uv run ruff check src/` and `uv run ruff format src/`.
