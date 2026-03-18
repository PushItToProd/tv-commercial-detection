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
  classify.py        Image classification logic (calls llama.cpp via OpenAI-compat API)
  config.py          App configuration dataclass
  main.py            App factory and lifespan startup
  state.py           In-memory application state
  matrix.py          HDMI matrix control helpers
  metrics.py         Prometheus metrics setup
  routes/            FastAPI routers (receive, review, status, trigger_matrix)
  prompt/            LLM prompt text files and images used for OpenCV-based classification
  frames/            Saved frames and labels (runtime, gitignored)
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
| `prometheus-fastapi-instrumentator` | Metrics endpoint (`/metrics`) |

### Configuration

Config is layered (later overrides earlier):
1. `server/config.json` (optional)
2. Environment variables: `DETECTOR_MATRIX_URL`, `DETECTOR_SAVE_DIR`, `DETECTOR_INCORRECT_DIR`, `DETECTOR_ENABLE_DEBOUNCE`
3. `LLAMA_SERVER_URL` — URL for the llama.cpp server (default: `http://192.168.1.27:3002`)
4. `PROMPT_FILE` — path to the classification prompt (default: `server/prompt/prompt.txt`)

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/receive` | Accept a screenshot + playback state from the extension |
| `GET/POST` | `/review` | Manual review UI for saved frames |
| `GET` | `/status` | SSE stream of current state (is_ad, paused, seeking) |
| `GET` | `/is_ad` | Simple HTML status page (used on secondary devices) |
| `POST` | `/trigger_matrix` | Manually trigger an HDMI matrix switch |

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

`classify.py` resizes each image to at most 800 px on its longest side, encodes it as JPEG (quality 50), and sends it to the llama.cpp server using the OpenAI vision API. The prompt lives in `server/prompt/prompt.txt`.

Classification labels: `ad`, `content` (racing), `unknown`.

---

## Development notes

- State is kept in a module-level `state` object in `state.py` — not persisted between restarts.
- There is currently no test suite.
- Type checking: the project uses Pyright (see inline `# pyright: ignore` comments).
- Linting/formatting: not yet configured — follow the style of surrounding code when making changes.
