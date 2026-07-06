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

This project was built around a Portta 4x2 HDMI multiviewer. The [`hdmi-matrix-control`](https://github.com/PushItToProd/hdmi-matrix-control) service communicates with it over a USB serial port exposed at `/dev/ttyACM0`. The matrix control API supports two outputs (`A`, `B`) and up to four inputs; the detector currently only switches between two inputs across two outputs.

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

### 2. Choose a model

The server requires a vision-capable model served by llama.cpp. Set `MODEL_FILE` and `MMPROJ_FILE` in `.env` to point to the model and its multimodal projector.

Tested models:
- **Qwen3 Omni 30B-A3B** — current recommendation; supports both images and audio, which enables `enable_llm_audio`
- **Qwen3.5 4B (Q4_K_M)** — image-only; good results at lower VRAM cost

Gemma 4 vision models were tried and produced poor classification results.

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

### 4. Start services

```bash
docker compose up -d
```

This starts:
- `llama` — llama.cpp with CUDA, serving a vision model
- `hdmi-matrix-control` — serial bridge for the HDMI matrix
- `receiver` — the FastAPI detector server

### 5. Load the browser extension

1. Go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on..."
3. Select any file inside `browser_extension/`
4. Open the extension popup, set the server URL to `http://localhost:11679/receive`, set the capture interval, and start capture

The capture interval controls how often a frame is sent to the server. The right value depends on your GPU — start around 3 seconds and watch `nvtop` to check whether inference is keeping up. The extension default of 10 seconds is too slow to reliably catch the start of a break.

### 6. (Optional) Install the native audio host

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

## Status pages

The server exposes two browser-accessible views useful for monitoring classification in real time:

- `http://localhost:11679/is_ad` — simple status page showing the current classification; designed to be opened on a secondary device
- `http://localhost:11679/review` — review UI for inspecting saved frames and their classifications

## Perceptual hash overrides

Behavior of both the OpenCV and LLM-based classification is inconsistent and sometimes produces erroneous results. The phash override system is an attempt to short-circuit this: when you flag an incorrect classification using the frontend UI (which invokes the `/flag_frames` endpoint), a perceptual hash of each labeled frame is stored with the corrected classification. On subsequent classifications, if an incoming frame's perceptual hash is within `phash_threshold` of any stored hash, the stored label is returned immediately without running the full pipeline.

In practice, this has had limited effectiveness. Commercials tend to have frequent visual changes, but the phash only matches images that are nearly identical, so unless the extension happens to screenshot almost the exact same part of the ad in the future, it's unlikely to match.

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
| `enable_llm_audio` | `false` | Include audio in LLM classification requests (experimental; requires a model that supports audio input, e.g. Qwen3 Omni) |
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

- Only tested with YouTube TV and HBO Max in Firefox on Linux (Pop OS).
- State is in-memory and lost on restart.
- HDMI matrix control is specific to one hardware model and only supports swapping between two mappings of inputs to outputs.
- Classifier logo assets and LLM prompts are hand-tailored to specific sports on specific networks. Adding new sports and networks currently requires adding new classifiers or refining existing ones. Changes in graphics packages are likely to break the hand-tailored classifiers.
- TV networks are surprisingly inconsistent with their graphics, so primitive classification can fail.
- Classification is not fully reliable and can produce just enough errors to be frustrating. Some OpenCV classifiers produce intermittent false positives/negatives while the LLM-based approach is also prone to error. By far the hardest challenge is handling ads for the sport you're watching: e.g. some ads show images of race cars or clips of race broadcasts -- they usually have some subtle hints they're an ad, but figuring out how to detect that with OpenCV is very hard, and the LLM almost always thinks it's a race. Conversely, it's hard to correctly classify non-racing segments during a broadcast (e.g. interviews with drivers or shots of the commentators in the booth).
