#!/usr/bin/env python3
"""
Native messaging host for TV Commercial Detector.

Maintains a rolling PCM buffer captured from the monitor source of the sink the
browser is playing to (PulseAudio/PipeWire). On request, returns the last N
milliseconds as a base64-encoded WAV file.

Capturing the *default* sink's monitor isn't good enough: audio moves between
sinks (a Bluetooth speaker connects and grabs the default, or the browser is
pinned to HDMI while the default is elsewhere) and a monitor of the wrong sink
is real but silent, so capture dies with no error anywhere. Instead the host
looks up the sink the browser's own stream feeds, and a watcher thread follows
it when it moves.

Firefox native messaging protocol: messages are framed with a 4-byte
little-endian length prefix on both stdin and stdout.

Environment variables:
  AUDIO_BUFFER_SECONDS  — max seconds to keep in the rolling buffer (default: 10)
  AUDIO_SAMPLE_RATE     — sample rate in Hz (default: 44100)
  AUDIO_CHANNELS        — number of channels (default: 1)
  AUDIO_DEVICE          — sounddevice input device name or index (default: None,
                          which auto-detects as described above; setting this
                          pins the device and disables detection entirely)
  AUDIO_STREAM_MATCH    — comma-separated substrings identifying the browser's
                          playback stream (default: "firefox")
  AUDIO_POLL_SECONDS    — how often to re-check where the browser is playing
                          (default: 5)
  AUDIO_SILENCE_WARN_SECONDS — warn if capture stays silent this long while the
                          browser is actively playing (default: 30)
"""

import argparse
import base64
import io
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from datetime import datetime

import sounddevice as sd
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE: int = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))
CHANNELS: int = int(os.environ.get("AUDIO_CHANNELS", "1"))
BUFFER_SECONDS: float = float(os.environ.get("AUDIO_BUFFER_SECONDS", "10"))
DEVICE: str | int | None = os.environ.get("AUDIO_DEVICE") or None
STREAM_MATCH: list[str] = [
    s.strip().lower()
    for s in os.environ.get("AUDIO_STREAM_MATCH", "firefox").split(",")
    if s.strip()
]
POLL_SECONDS: float = float(os.environ.get("AUDIO_POLL_SECONDS", "5"))
SILENCE_WARN_SECONDS: float = float(os.environ.get("AUDIO_SILENCE_WARN_SECONDS", "30"))
# Repeat interval for the "capture is silent" warning, so a whole race of
# silence doesn't fill the log.
SILENCE_REWARN_SECONDS: float = 60.0

# sounddevice uses int16 frames; numpy dtype to match
DTYPE = "int16"

# ── logging (stderr only — stdout is reserved for native messaging) ────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[audio_capture] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── monitor source detection ──────────────────────────────────────────────────

def _pactl(*args: str) -> str | None:
    """Run a pactl subcommand and return stdout, or None if it fails."""
    try:
        proc = subprocess.run(
            ["pactl", *args],
            capture_output=True,
            text=True,
            timeout=3,
            # pactl translates its field labels; the parsers below match English.
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("pactl %s failed: %s", " ".join(args), exc)
        return None
    if proc.returncode != 0:
        logger.debug("pactl %s exited %d: %s", " ".join(args), proc.returncode, proc.stderr.strip())
        return None
    return proc.stdout


def _sink_names() -> dict[str, str]:
    """Map sink index → sink name (sink-inputs reference sinks by index)."""
    out = _pactl("list", "sinks", "short")
    names: dict[str, str] = {}
    for line in (out or "").splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            names[fields[0].strip()] = fields[1].strip()
    return names


def _default_sink() -> str | None:
    out = _pactl("info")
    for line in (out or "").splitlines():
        if line.startswith("Default Sink:"):
            return line.split(":", 1)[1].strip()
    return None


def _monitor_exists(monitor: str) -> bool:
    out = _pactl("list", "sources", "short")
    return any(monitor in line for line in (out or "").splitlines())


@dataclass
class PlaybackStream:
    """One entry from `pactl list sink-inputs`."""

    sink_index: str
    app: str
    corked: bool  # a corked stream is connected but not currently playing


def _playback_streams() -> list[PlaybackStream]:
    out = _pactl("list", "sink-inputs")
    if out is None:
        return []

    streams: list[PlaybackStream] = []
    sink_index = ""
    corked = False
    names: list[str] = []

    def flush() -> None:
        if sink_index:
            streams.append(
                PlaybackStream(sink_index=sink_index, app=" ".join(names), corked=corked)
            )

    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Sink Input #"):
            flush()
            sink_index, corked, names = "", False, []
        elif stripped.startswith("Sink:"):
            sink_index = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Corked:"):
            corked = stripped.split(":", 1)[1].strip().lower() == "yes"
        elif "=" in stripped:
            key, _, value = stripped.partition("=")
            # These three between them identify the owning app across the
            # variations in how browsers label their streams.
            if key.strip() in (
                "application.name",
                "application.process.binary",
                "media.name",
            ):
                names.append(value.strip().strip('"'))
    flush()
    return streams


@dataclass
class Target:
    """Where we should be capturing from, and why."""

    monitor: str | None  # monitor source name; None means "use the default input"
    reason: str  # for logging
    playing: bool = False  # the browser has an uncorked stream right now


def resolve_target() -> Target:
    """Pick the monitor source to capture, preferring the browser's own sink.

    Falls back to the default sink's monitor when the browser isn't playing
    (it may not have started yet), and finally to the default input device if
    pactl isn't available at all.
    """
    sinks = _sink_names()
    browser = [
        s
        for s in _playback_streams()
        if any(m in s.app.lower() for m in STREAM_MATCH) and s.sink_index in sinks
    ]
    # An uncorked stream is the one actually producing audio; a corked one still
    # tells us where playback will land when it resumes.
    for stream in sorted(browser, key=lambda s: s.corked):
        monitor = f"{sinks[stream.sink_index]}.monitor"
        if _monitor_exists(monitor):
            return Target(
                monitor=monitor,
                reason=f"browser stream {stream.app!r} on {sinks[stream.sink_index]}"
                + ("" if not stream.corked else " (paused)"),
                playing=not stream.corked,
            )

    default_sink = _default_sink()
    if default_sink:
        monitor = f"{default_sink}.monitor"
        if _monitor_exists(monitor):
            return Target(
                monitor=monitor,
                reason="default sink (no browser playback found)",
            )
    return Target(monitor=None, reason="pactl unavailable; using default input device")


# ── rolling PCM buffer ────────────────────────────────────────────────────────

# Each entry is a numpy array of shape (n_frames, CHANNELS) with dtype int16.
# Total frames kept = BUFFER_SECONDS * SAMPLE_RATE.
_MAX_FRAMES = int(BUFFER_SECONDS * SAMPLE_RATE)
_buf: deque[np.ndarray] = deque()
_buf_frames: int = 0  # total frames currently held across all chunks
_buf_lock = threading.Lock()

# Monotonic time of the last block that carried any signal, so the watcher can
# tell a live source from a monitor of the wrong sink.
_last_signal_at: float = time.monotonic()


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """Called by sounddevice on each audio block; appends a copy to the buffer."""
    if status:
        logger.warning("sounddevice status: %s", status)

    chunk = indata.copy()  # indata is a view; copy before releasing
    if chunk.size and int(np.abs(chunk).max()) > 0:
        global _last_signal_at
        _last_signal_at = time.monotonic()

    with _buf_lock:
        global _buf_frames
        _buf.append(chunk)
        _buf_frames += frames

        # Trim oldest chunks until we're within the max buffer size
        while _buf_frames > _MAX_FRAMES and _buf:
            oldest = _buf.popleft()
            _buf_frames -= len(oldest)


def _clear_buffer() -> None:
    """Drop buffered audio — used when the capture source changes, so a clip
    can't straddle two sources (or trail silence from a dead one)."""
    global _buf_frames
    with _buf_lock:
        _buf.clear()
        _buf_frames = 0


def _get_audio_wav(duration_ms: int) -> bytes:
    """
    Slice the last `duration_ms` milliseconds from the PCM buffer and return
    the data encoded as a WAV file (bytes).
    """
    want_frames = int(duration_ms / 1000 * SAMPLE_RATE)

    with _buf_lock:
        if _buf_frames == 0:
            # Return a silent WAV of the requested duration
            pcm = np.zeros((want_frames, CHANNELS), dtype=DTYPE)
        else:
            # Concatenate all chunks, then take the last want_frames
            all_frames = np.concatenate(list(_buf), axis=0)
            pcm = all_frames[-want_frames:]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── capture stream management ─────────────────────────────────────────────────

class MonitorCapture:
    """Keeps an input stream bound to the monitor source we currently want.

    The target moves at runtime — the browser may not be playing when the host
    starts, and sinks come and go — so the stream has to be reopened rather than
    chosen once at startup.
    """

    def __init__(self) -> None:
        self._stream: sd.InputStream | None = None
        self._source: str | None = None
        self._bound = False
        self._lock = threading.Lock()

    @property
    def source(self) -> str | None:
        """Monitor source currently captured; None means the default input."""
        return self._source

    @property
    def bound(self) -> bool:
        """Whether a stream is currently open."""
        return self._bound

    def bind(self, source: str | None) -> bool:
        """(Re)open the stream against *source*. Returns False if it can't open."""
        with self._lock:
            if self._bound and source == self._source:
                return True

            stream = None
            try:
                # The ALSA pulse plugin reads PULSE_SOURCE when the device is
                # opened, so this has to be set before constructing the stream.
                if source:
                    os.environ["PULSE_SOURCE"] = source
                else:
                    os.environ.pop("PULSE_SOURCE", None)
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    device="pulse" if source else DEVICE or "default",
                    callback=_audio_callback,
                )
                stream.start()
            except Exception:
                logger.exception("Could not open audio stream for source %r", source)
                if stream is not None:
                    stream.close()
                return False

            old = self._stream
            self._stream, self._source, self._bound = stream, source, True
            if old is not None:
                try:
                    old.stop()
                    old.close()
                except Exception:
                    logger.exception("Error closing the previous audio stream")
            _clear_buffer()
            global _last_signal_at
            _last_signal_at = time.monotonic()
            logger.info("Capturing from %s", source or DEVICE or "default input device")
            return True

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                self._bound = False


def _watch_target(capture: MonitorCapture, stop: threading.Event) -> None:
    """Follow the browser's playback sink and warn when capture goes silent."""
    last_warned = 0.0
    while not stop.wait(POLL_SECONDS):
        try:
            target = resolve_target()
            if target.monitor != capture.source or not capture.bound:
                logger.info("Audio target changed → %s", target.reason)
                capture.bind(target.monitor)

            # Only a stream that is actually playing makes silence a fault; a
            # paused video is legitimately silent.
            silent_for = time.monotonic() - _last_signal_at
            now = time.monotonic()
            if (
                target.playing
                and silent_for > SILENCE_WARN_SECONDS
                and now - last_warned > SILENCE_REWARN_SECONDS
            ):
                last_warned = now
                logger.warning(
                    "No audio signal for %.0fs while the browser is playing"
                    " (capturing %s). The monitor source may not match the sink"
                    " the browser feeds.",
                    silent_for,
                    capture.source or "default input device",
                )
        except Exception:
            logger.exception("Error while checking the audio capture target")


def _status(capture: MonitorCapture) -> dict:
    """Snapshot of what the host is capturing — for `status` and diagnostics."""
    target = resolve_target()
    with _buf_lock:
        buffered = _buf_frames / SAMPLE_RATE
    return {
        "source": capture.source,
        "buffered_seconds": round(buffered, 2),
        "silent_seconds": round(time.monotonic() - _last_signal_at, 1),
        "target": target.reason,
        "browser_playing": target.playing,
        "target_matches": target.monitor == capture.source,
    }


# ── native messaging I/O ──────────────────────────────────────────────────────

def _read_message() -> dict:
    """Read one length-prefixed JSON message from stdin."""
    raw_len = sys.stdin.buffer.read(4)
    if len(raw_len) < 4:
        raise EOFError("stdin closed")
    (length,) = struct.unpack("<I", raw_len)
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def _write_message(obj: dict) -> None:
    """Write one length-prefixed JSON message to stdout."""
    payload = json.dumps(obj).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("<I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


# ── save loop (optional, for standalone testing) ─────────────────────────────

def _save_loop(save_dir: str, frequency: float, duration_s: float, stop: threading.Event) -> None:
    """Periodically save a WAV snapshot to *save_dir* until *stop* is set."""
    os.makedirs(save_dir, exist_ok=True)
    while not stop.wait(frequency):
        duration_ms = int(duration_s * 1000)
        try:
            wav_bytes = _get_audio_wav(duration_ms)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"audio_{timestamp}.wav")
            with open(path, "wb") as f:
                f.write(wav_bytes)
            logger.info("Saved %s", path)
        except Exception:
            logger.exception("Error saving audio snapshot")


# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save-dir", metavar="DIR", help="Save .wav snapshots to this directory")
    parser.add_argument("--frequency", type=float, default=10.0, metavar="SECS", help="Seconds between saved snapshots (default: 10)")
    parser.add_argument("--duration", type=float, default=4.0, metavar="SECS", help="Duration of each saved snapshot in seconds (default: 4)")
    # Firefox passes these two arguments when launching the native host; ignore them but allow them for compatibility.
    parser.add_argument("app_manifest", nargs="?", help="(Firefox) Path to the native messaging host manifest JSON file")  # for compatibility with Firefox, which passes this as an argument
    parser.add_argument("extension_id", nargs="?", help="(Firefox) The ID of the calling extension")  # for compatibility with Firefox
    args = parser.parse_args()

    logger.info(
        "Starting — rate=%d Hz, channels=%d, buffer=%gs, device=%r",
        SAMPLE_RATE, CHANNELS, BUFFER_SECONDS, DEVICE,
    )

    capture = MonitorCapture()
    stop_event = threading.Event()

    if DEVICE is not None:
        # An explicit device pins capture; detection and the watcher are skipped.
        logger.info("AUDIO_DEVICE is set — capturing from %r without detection", DEVICE)
        if not capture.bind(None):
            logger.error("Could not open the configured audio device — exiting")
            return
    else:
        target = resolve_target()
        logger.info("Audio target: %s", target.reason)
        if target.monitor is None:
            logger.warning(
                "Could not detect a monitor source; falling back to the default"
                " input device — captured audio may be silent"
            )
        capture.bind(target.monitor)
        threading.Thread(
            target=_watch_target, args=(capture, stop_event), daemon=True
        ).start()

    try:
        if args.save_dir:
            save_thread = threading.Thread(
                target=_save_loop,
                args=(args.save_dir, args.frequency, args.duration, stop_event),
                daemon=True,
            )
            save_thread.start()
            logger.info(
                "Saving %.1fs snapshots to %r every %.1fs",
                args.duration, args.save_dir, args.frequency,
            )
            # Standalone mode: just wait until interrupted; no stdin to read.
            logger.info("Ready — press Ctrl-C to stop")
            try:
                stop_event.wait()
            except KeyboardInterrupt:
                logger.info("Interrupted — exiting")
        else:
            logger.info("Ready — waiting for messages")
            while True:
                try:
                    msg = _read_message()
                except EOFError:
                    logger.info("stdin closed — exiting")
                    break

                command = msg.get("command")

                if command == "get_audio":
                    duration_ms = int(msg.get("duration_ms", 4000))
                    logger.info("get_audio request: duration_ms=%d", duration_ms)
                    try:
                        wav_bytes = _get_audio_wav(duration_ms)
                        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                        _write_message({"audio": audio_b64, "source": capture.source})
                    except Exception as exc:
                        logger.exception("Error encoding audio")
                        _write_message({"error": str(exc)})

                elif command == "status":
                    _write_message(_status(capture))

                elif command == "ping":
                    _write_message({"pong": True})

                else:
                    logger.warning("Unknown command: %r", command)
                    _write_message({"error": f"unknown command: {command!r}"})
    finally:
        stop_event.set()
        capture.close()


if __name__ == "__main__":
    main()
