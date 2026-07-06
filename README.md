# tv-commercial-detector

Detects TV commercial breaks during live NASCAR broadcasts on YouTube TV and switches an HDMI matrix to an alternate input for the duration of the break.

## How it works

A Firefox extension periodically captures a screenshot (and optionally a short audio clip) of the active browser tab and POSTs it to a local server. The server classifies the frame using a pipeline of OpenCV template matching and, as a fallback, a locally-hosted multimodal LLM. When the classification changes between `content` and `ad`, the server sends switch commands to a small HTTP service that controls an HDMI matrix over a serial port.

## Components

| Component | Description |
|---|---|
| `browser_extension/` | Firefox (Manifest V2) extension. Captures screenshots at a configurable interval and POSTs them to the server. Optionally requests system audio from the native host. |
| `native_host/` | Firefox native messaging host. Captures audio from the PulseAudio/PipeWire monitor source and streams it to the extension on demand. |
| `server/` | FastAPI app. Classifies frames, tracks broadcast state, and triggers HDMI matrix switches. |
| `hdmi-matrix-control` | Separate project (external). HTTP-to-serial bridge for the HDMI matrix. Configured via `HDMI_MATRIX_CONTROL_DIR` in `.env`. |

## Requirements

- Linux with PulseAudio or PipeWire (for audio capture via the native host)
- Firefox
- Docker and Docker Compose (to run the LLM server and HDMI matrix control service)
- NVIDIA GPU with CUDA support (for the llama.cpp container; see `docker-compose.yml`)
- Python ≥ 3.14 and `uv` (for running the server outside Docker, or for development)
- A compatible HDMI matrix with serial control (see [Hardware](#hardware))

## Hardware

This project was built around a specific HDMI matrix. The `hdmi-matrix-control` service communicates with it over a USB serial port exposed at `/dev/ttyACM0`. The matrix control API supports two outputs (`A`, `B`) and up to four inputs; the detector currently only switches between two inputs across two outputs.

**The HDMI matrix control component is not generic.** It will not work with other hardware without writing a compatible `hdmi-matrix-control` service that exposes the same HTTP API (`POST /set-output-input`).

## Setup

### 1. Clone and configure

```bash
cp example.env .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `LLAMA_PORT` | Port for the llama.cpp container |
| `MODELS_HOST_DIR` | Host path containing `.gguf` model files |
| `MODEL_FILE` | Vision model filename (relative to `MODELS_HOST_DIR`) |
| `MMPROJ_FILE` | Multimodal projector filename |
| `HDMI_MATRIX_CONTROL_DIR` | Path to the `hdmi-matrix-control` project |
| `HDMI_SERIAL_PORT` | Serial device for the matrix (e.g., `/dev/ttyACM0`) |
| `HDMI_MATRIX_CONTROL_PORT` | Port to expose the matrix control service on |
| `RECEIVER_PORT` | Port for the detector server (default: `11679`) |

### 2. Start services

```bash
docker compose up -d
```

This starts:
- `llama` — llama.cpp with CUDA, serving a vision model
- `hdmi-matrix-control` — serial bridge for the HDMI matrix
- `receiver` — the FastAPI detector server

### 3. Configure the server

Copy or edit `server/config.json` to set `output_settings`, which maps classification results to HDMI inputs:

```json
{
  "output_settings": {
    "ad": { "A": "1", "B": "4" },
    "content": { "A": "4", "B": "1" }
  }
}
```

Outputs are `"A"` and `"B"`. Input values are integers (as strings). Adjust to match your physical wiring.

Also set `classifier_profile` to match the broadcast you're watching (see [Classifier Profiles](#classifier-profiles)).

### 4. Load the browser extension

1. Go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on..."
3. Select any file inside `browser_extension/`
4. Open the extension popup, set the server URL to `http://localhost:11679/receive`, and start capture

### 5. (Optional) Install the native audio host

Required only if `enable_llm_audio` is enabled in the server config.

```bash
cd native_host
./install.sh
```

Re-run `install.sh` if the repo is moved; it writes the absolute path to the `run` wrapper into `~/.mozilla/native-messaging-hosts/`. Reload the extension in Firefox after installing.

## Classifier profiles

Set `classifier_profile` in `server/config.json` (or `DETECTOR_CLASSIFIER_PROFILE` env var).

| Profile | Broadcast | Notes |
|---|---|---|
| `nascar_on_fox` | NASCAR on Fox/FS1 | Logo detection → rectangle detection → LLM quick check → LLM full prompt |
| `nhra_on_fox` | NHRA drag racing on Fox/FS1 | Same pipeline as `nascar_on_fox` with NHRA-specific logo assets |
| `nascar_on_hbo_max` | NASCAR on TNT Sports/HBO Max | OpenCV only; no LLM pass needed |

HBO Max does not insert traditional ad breaks. Instead, TNT Sports shows either a full-screen "we'll be right back" card (classified as `ad`, triggering a switch) or a side-by-side "commercial break in progress" overlay where live racing is still visible (classified as `content`, no switch). Everything else defaults to `content`. The signals are reliable enough that an LLM fallback is unnecessary.

## Configuration reference

`server/config.json` (overrides defaults; all fields optional):

| Field | Default | Description |
|---|---|---|
| `matrix_url` | `http://localhost:5000` | URL for the `hdmi-matrix-control` service |
| `llm_url` | `http://localhost:3002` | URL for the llama.cpp server |
| `save_dir` | `frames` | Directory for saved frames |
| `classifier_profile` | `nascar_on_fox` | Active classifier |
| `enable_debounce` | `false` | Require two consecutive matching results before switching |
| `auto_switch` | `true` | Enable automatic HDMI switching |
| `enable_llm_audio` | `false` | Include audio in LLM classification requests |
| `phash_threshold` | `10` | Max perceptual hash distance for override matches |
| `output_settings` | `{}` | Input/output mapping per classification (see above) |

Environment variables (`DETECTOR_MATRIX_URL`, `DETECTOR_SAVE_DIR`, `DETECTOR_ENABLE_DEBOUNCE`, `DETECTOR_CLASSIFIER_PROFILE`, `LLAMA_SERVER_URL`) override `config.json`.

## Development

```bash
cd server

# Install dependencies
uv sync

# Run dev server
uv run uvicorn tv_commercial_detector.main:create_app --factory --reload --host 0.0.0.0 --port 11679

# Lint / format
uv run ruff check src/
uv run ruff format src/

# Run unit tests (integration tests require a live llama.cpp server)
uv run pytest tests/ -m "not integration"
```

Python ≥ 3.14 required.

## Known limitations

- Only tested with YouTube TV in Firefox on Linux
- HDMI matrix control is specific to one hardware model
- Only two inputs and two outputs are used
- Classifier logo assets are hand-tuned to specific broadcast overlays and may need updating across seasons or network rebrands
- State is in-memory and lost on restart
