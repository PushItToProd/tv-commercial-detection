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
    audio_health.py    Detects silent (dead) audio capture from the received clips
    classify.py        Entry point for classification; dispatches to active classifier profile
    config.py          App configuration dataclass (AppConfig)
    frame_saver.py     Periodic frame saving; rolling in-memory buffer of recent frames
    main.py            App factory and lifespan startup
    matrix.py          HDMI matrix control helpers
    metrics.py         Prometheus metrics setup
    phash_override.py  Perceptual hash override system (manual label corrections)
    state.py           In-memory application state (AppState)
    video_timebase.py  Parses the player's timebase fields (see below)
    classification/    Low-level classification primitives
      llm_match.py     OpenAI-compat LLM calls (image resize, prompt, response parsing)
      logo_match.py    OpenCV template matching for network/side-by-side logos
      rectangle_match.py  OpenCV contour detection for known ad-break bounding boxes
      result.py        ClassificationResult dataclass
    classifiers/       Pluggable classifier profiles (selected via classifier_profile config)
      nascar_on_fox.py Multi-pass classifier: logo → rectangle → LLM quick check → LLM prompt
      nhra_on_fox.py   Variant for NHRA drag racing broadcasts on Fox/FS1
      nascar_on_hbo_max.py  TNT Sports coverage on HBO Max; OpenCV logo checks only, no LLM pass needed
      nascar_on_nbc.py  NBC Sports Cup coverage on NBC and USA; peacock/USA bug checks, then LLM fallback
    routes/            FastAPI routers
    prompt/            LLM prompt text and logo images used for OpenCV matching
    static/            Static assets (Bootstrap CSS/JS for UI templates)
    templates/         Jinja2 templates (review UI, is_ad page)
  tests/               Unit and integration tests
    classification/    Tests for logo_match and rectangle_match
    routes/            Tests for each route
    integration/       Integration tests (require a live llama.cpp server)
  scripts/             Utility scripts (record_broadcast.py, find_dupes.py, view_classification_results.py, etc.)
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

### Recording a whole broadcast

`scripts/record_broadcast.py` is a standalone receiver that archives every frame
and audio clip it's sent, with no classification, matrix switching or debounce.
The extension posts to every endpoint in its list, so it can run alongside the
detector — add its `/receive` URL as a second endpoint in the popup.

```bash
uv run python scripts/record_broadcast.py -d /mnt/data/tv-commercial-detector/full_broadcast_frames
```

Each broadcast gets its own directory under `-d`, named from the page hostname,
the network name (only YouTube TV reports one) and the video title — falling
back to the page title on sites that don't report a video title:

```
<out-dir>/tv.youtube.com/Oregon-s_FOX_Autotrader_400/
<out-dir>/play.hbomax.com/eero_400_-_HBO_Max/
```

Those directories use the same layout as `save_dir` (`images/`, `audio/`,
`classifications.jsonl`) and the frame saver's filename convention, so a
recording can be reviewed by pointing `DETECTOR_SAVE_DIR` at it or fed to
`dedupe_frames.py`. A title change mid-stream (pre-race show → race) opens a new
directory; returning to a title already seen resumes appending to its directory.
`GET /status` reports what's been written so far, and Ctrl-C prints a summary.

### The player's timebase

Every frame records where the player was, not just when the frame was grabbed.
`video_offset` is the player's `currentTime`, and the timebase fields say what
it is measured against:

| Field | Meaning |
|---|---|
| `video_id` | The program, parsed from the page URL (`/watch/<id>` or `?v=<id>`) |
| `video_duration` | Length in seconds, or `null` when not finite |
| `is_live` | `true` live, `false` a recording, `null` not yet known |
| `seekable_start` / `seekable_end` | Bounds of the seekable range |

This matters because `currentTime` is only comparable across separate capture
passes when it counts from the start of the program rather than from when the
player loaded. A recording reports a finite duration and a seekable range
starting at 0; a live stream reports an infinite duration and a DVR window whose
start creeps forward. Recording the distinction per frame is what makes it
possible to treat two discontinuous passes over one program — capture, reboot,
capture again the next day — as a single timeline, and to line up passes taken
at different capture intervals.

Offsets are a coordinate, not a key: capture jitter means two passes never land
on identical values (observed inter-frame deltas run 1.60–2.11 s against a
nominal 2.0), and a backward seek can put two frames at the same offset. Joining
passes is a nearest-neighbor match within a tolerance, and the filename stays
the stable per-frame identifier.

`video_duration` is split into a number plus `is_live` because a live stream's
duration is `Infinity`, which has no JSON representation — writing it bare
produces a `classifications.jsonl` that Python reads back but `jq` and
`JSON.parse` reject. `video_timebase.py` holds the parsing, shared by the
detector's `/receive` and `record_broadcast.py` so both write the same fields.
Every field is independently optional, so an older extension still posts
successfully and simply records nulls.

### Pruning silent audio files

`scripts/prune_silent_audio.py` removes audio clips whose peak amplitude is at
or below `audio_silence_threshold` — the residue of capture bound to a sink the
browser wasn't playing to. It touches only `audio/`; clips carry no metadata
records of their own, so a removed clip dangles nothing and its frame stays. The
dry run prints a per-day table, which is the quickest way to see when capture
died and whether it has recovered.

```bash
uv run python scripts/prune_silent_audio.py            # dry run + per-day report
uv run python scripts/prune_silent_audio.py --apply
```

Additional `AppConfig` fields:
- `phash_threshold` — max perceptual hash distance for override matches (default: `10`)
- `enable_llm_audio` — enable audio-based LLM classification (default: `false`)
- `llm_model_name` — model name sent to llama.cpp, in case we're using llama.cpp in router mode (default: `LLAMA_MODEL_NAME` env var, or `"local"`)
- `audio_silence_threshold` — peak amplitude (fraction of full scale) below which a clip counts as silent (default: `0.001`)
- `audio_silence_clips` — consecutive silent clips before audio capture is called dead (default: `3`)
- `video_report_stale_seconds` — seconds without a report from the extension before its last reading is called stale; `0` disables the check (default: `30.0`)

### Audio health

Dead audio capture is silent in both senses: clips keep arriving at the right
length and cadence carrying nothing but zeros, and nothing downstream complains
— the LLM just gets a silent clip and the save dir fills with useless WAVs. An
entire summer of collection was lost this way before anyone noticed.

`audio_health.py` measures the peak amplitude of every clip received by
`/receive`. After `audio_silence_clips` consecutive silent ones it logs a
warning (once on the transition, then at most every 5 minutes) and reports the
condition in `/is_ad/status` as `audio_warning`, which the `/is_ad` page renders
as a banner. Signal on any clip clears it.

Frames arriving without audio are not reported: the extension only sends clips
while the native host is connected, and running without it isn't a fault. Clips
that can't be parsed as WAV neither start nor break a silent streak — a corrupt
clip says nothing about whether the capture source is live.

### Video reporting status

`state.paused` and `state.seeking` are the last reading the extension sent, and
on their own they can't say whether that reading still describes anything. The
server holds them until something replaces them, so a stopped extension, a
closed tab, a page with no player on it and a genuinely paused video all present
identically — as `paused`, which is also the startup default. That ambiguity is
what made a stuck capture look like a paused video for as long as it took to
notice by hand.

`AppState.video_status()` collapses the flags and the report age into one value,
and is what the `/is_ad` page renders. Most severe first:

| Status | Meaning |
|---|---|
| `waiting` | Nothing has ever reported; `paused` is a default, not a reading |
| `stale` | Reported once, but not within `video_report_stale_seconds` |
| `no_video` | Extension is reporting and can't find a player element |
| `paused` / `seeking` / `playing` | A current reading |

`no_video` is reported by the extension (a `no_video` form field on `/receive`
and `/video-state`); `waiting` and `stale` are inferred from `last_report_at`,
which every call to either endpoint refreshes. Ordering is by severity because a
reading nobody has confirmed recently says nothing about the player regardless
of what it holds — `stale` therefore outranks `no_video`, and `paused` outranks
`seeking` to match what the page has always shown for a scrub on a paused video.

`/is_ad/status` carries `report_age` and `stale_after_seconds` alongside
`video_status` so the page can age a reading into `stale` on a local timer. It
has to be able to: nothing arriving is the whole signal, so there is no push
coming to announce it.

None of this gates matrix switching. Without frames there is nothing to
classify and no switch to make, so the status is purely diagnostic — it exists
so the `/is_ad` page stops showing a confident answer it has no basis for.

### Routes

| Method | Path | Description |
|---|---|---|
| `POST` | `/receive` | Accept a screenshot + playback state from the extension |
| `POST` | `/video-state` | Update playback state only (no image), including `no_video` |
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
  - `content_scripts/track_interactions.js` — injected into every page; decides which `<video>` is the player and reports its play/pause/seek events back to the background script.
  - `content_scripts/get_video_bounds.js` — injected on demand; reads the tracked element's playing state, timebase and bounding rect.
  - `popup.html` / `popup.js` — configuration UI (server URL, capture interval, start/stop).

Configuration (server endpoint URL, capture interval) is stored via `browser.storage.local`.

### Choosing the page's player element

A page routinely holds several `<video>` elements — the main player, a muted
browse-row preview, an ad slot that never plays — and which one is the real
player is not decidable when the content script runs. Resolving it once at
`document_idle` is unrecoverable in both directions: a player that hasn't
loaded metadata (nothing has been played in the tab yet) can't be recognized at
all, and an element that never plays reports `paused` forever while the real
video runs beside it. Neither has a repair path, because a MutationObserver
only ever sees *newly added* nodes.

So `resolve()` is called on every capture tick, on the observer, and on a
`RESCAN_MS` timer. It keeps the current element while it stays in the document,
and re-ranks only when there is none — playing beats loaded-metadata beats
never-loaded, ties broken by rendered area. Ownership otherwise moves only when
another element fires `play` while the current one is paused, ended or
detached; starting playback is the one unambiguous signal of which element the
user is watching.

Listeners are attached to every `<video>` seen and never removed. Events are
instead ignored unless they come from the current element, which is what keeps
a still-playing preview from speaking for a player the user paused, and
suppresses the `pause` an element fires when it's removed from the document —
reported, that would tell the server the video stopped when it hadn't.

Sticky state gets a timeout for the same reason: `isSeeking` is cleared after
`SEEK_TIMEOUT_MS` because an element torn down mid-seek never fires `seeked`,
and a latched flag suppresses every subsequent capture.

A tick that finds no player element POSTs `no_video=true` to `/video-state`
instead of going quiet, and logs its consecutive count in the popup. See "Video
reporting status" for why the server can't infer that condition on its own.

---

## Native host

`native_host/audio_capture.py` is a Firefox [native messaging](https://developer.mozilla.org/en-US/docs/Mozilla/Add-ons/WebExtensions/Native_messaging) host that gives the browser extension access to system audio.

- **Protocol** — Firefox launches the host process and communicates via stdin/stdout using 4-byte little-endian length-prefixed JSON messages.
- **Audio capture** — Opens a `sounddevice.InputStream` against the PulseAudio/PipeWire monitor source of the sink **the browser is playing to**. Incoming PCM is stored in a thread-safe rolling deque capped at `AUDIO_BUFFER_SECONDS` (default: 10 s).
- **Commands** accepted from the extension:
  - `get_audio` — returns the last `duration_ms` milliseconds of audio as a base64-encoded WAV in `{"audio": "...", "source": "<monitor source>"}`.
  - `status` — returns the current source, seconds buffered, seconds since the last non-zero sample, and where the browser is playing. Use it to diagnose silent capture.
  - `ping` — responds with `{"pong": true}` (health check).
- **Standalone mode** — pass `--save-dir DIR` to periodically write WAV snapshots to disk for testing without the extension.

### Following the browser's sink

Monitoring the *default* sink is not sufficient: sinks change under the host
(a Bluetooth speaker connects and takes the default, or the browser stays pinned
to HDMI while the default moves elsewhere) and the monitor of a sink nothing
plays to is a perfectly healthy source that returns pure digital silence. That
failure is invisible — clips keep arriving at the right length and cadence, full
of zeros.

So `resolve_target()` looks through `pactl list sink-inputs` for a stream whose
`application.name` / `application.process.binary` / `media.name` matches
`AUDIO_STREAM_MATCH`, and captures that sink's monitor, preferring an uncorked
(actually playing) stream over a corked one. It falls back to the default sink's
monitor when the browser isn't playing — it may not have started yet — and to
the default input device if `pactl` isn't available at all. A watcher thread
re-resolves every `AUDIO_POLL_SECONDS` and rebinds the stream when the target
moves, clearing the buffer so no clip straddles two sources.

Setting `AUDIO_DEVICE` pins the device and disables detection and the watcher
entirely.

The host also warns when it has seen no non-zero sample for
`AUDIO_SILENCE_WARN_SECONDS` *while the browser stream is uncorked* — a paused
video is legitimately silent, so the uncorked check is what keeps the warning
meaningful.

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
| `AUDIO_DEVICE` | auto-detected | `sounddevice` input device name or index; setting it disables sink detection |
| `AUDIO_STREAM_MATCH` | `firefox` | Comma-separated substrings identifying the browser's playback stream |
| `AUDIO_POLL_SECONDS` | `5` | How often to re-check where the browser is playing |
| `AUDIO_SILENCE_WARN_SECONDS` | `30` | Warn after this long with no signal while the browser is playing |

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

The `nascar_on_hbo_max` profile covers TNT Sports coverage on HBO Max. It runs two OpenCV logo checks — a full-screen "we'll be back" card (`ad`) and a side-by-side "commercial break in progress" overlay (`content`, since racing is still shown side-by-side) — and falls back to `content` by default. HBO Max doesn't insert traditional ad breaks, so those two cards are the only break signals there are; they're reliable enough that the rectangle-detection and LLM passes the other profiles need aren't necessary here.

The `nascar_on_nbc` profile covers NBC Sports Cup coverage on **both NBC and USA Network** — the same production and the same "NASCAR NON STOP" break, only the corner bug differs. It checks the side-by-side banner in the upper left (`ad`), then either network bug in the upper right (`content`), then falls through to the LLM quick check and `prompt_nbc.txt`.

The two bugs need different matching and are not interchangeable:

- **NBC peacock** — opaque and coloured, so matched **in colour**. `load_masked` / `mask_non_white` zero out everything that isn't near-white and would erase a six-colour logo entirely, so this template is loaded with a plain `cv2.imread`. Its search window is tight (`x 1740–1880, y 40–140` at 1920×1080); over the wide upper-right region the Fox profile uses, the weakest true positive scores below the strongest false positive and the match is unusable. Threshold `0.55`, measured at 21/21 recall with 0/3000 false positives.
- **USA wordmark** — white, so white-masked like the Fox logo, but *translucent*. Over a blown-out sky it fades to a near-invisible ghost and the masked region saturates into a uniform patch, where `TM_CCOEFF_NORMED` divides by zero and can report a perfect `1.0`. The mask-fraction guard in `usa_score` is what makes the check safe; without it every bright sky reads as `content`. Those ghost frames are deliberately not chased — they carry too little signal to reach without wrecking precision, and fall through to the LLM. Threshold `0.65`, measured at ~82% recall with 0/3000 false positives.

The side-by-side half is **unvalidated** — no NBC or USA ad-break frame with the banner exists in the dataset — and is guarded by `ENABLE_SIDE_BY_SIDE_CHECK`. Each bug check has its own toggle (`ENABLE_PEACOCK_CHECK`, `ENABLE_USA_CHECK`).

Note that the LLM fallback is weak on pre-race paddock and driver-intro content, which `_report_racing_related` tends to reject as an ad. The corner-bug checks run first specifically because they are far more reliable for that material.

Images are resized to at most 800 px on the longest side and JPEG-encoded (quality 50) before being sent to the LLM. The default prompt lives in `server/prompt/prompt.txt`; profiles can supply their own by passing `prompt=` to `llm_match.classify_by_prompt`.

Classification labels: `ad`, `content` (racing), `unknown`.

---

## Development notes

- State is kept in a module-level `state` object in `state.py` — not persisted between restarts.
- A test suite exists under `server/tests/`. Run unit tests with `uv run pytest tests/ -m "not integration"`. Integration tests (in `tests/integration/`) require a live llama.cpp server.
- Type checking: the project uses Pyright (see inline `# pyright: ignore` comments).
- Linting/formatting: configured via `ruff` in `pyproject.toml`. Run `uv run ruff check src/` and `uv run ruff format src/`.
