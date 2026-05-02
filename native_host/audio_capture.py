#!/usr/bin/env python3
"""
Native messaging host for TV Commercial Detector.

Maintains a rolling PCM buffer captured from the default system audio monitor
source (PulseAudio/PipeWire). On request, returns the last N milliseconds as a
base64-encoded WAV file.

Firefox native messaging protocol: messages are framed with a 4-byte
little-endian length prefix on both stdin and stdout.

Environment variables:
  AUDIO_BUFFER_SECONDS  — max seconds to keep in the rolling buffer (default: 10)
  AUDIO_SAMPLE_RATE     — sample rate in Hz (default: 44100)
  AUDIO_CHANNELS        — number of channels (default: 1)
  AUDIO_DEVICE          — sounddevice input device name or index (default: None,
                          which uses the system default — typically the monitor
                          source on PulseAudio/PipeWire)
"""

import base64
import io
import json
import logging
import os
import struct
import sys
import threading
import wave
from collections import deque

import sounddevice as sd
import numpy as np

# ── configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE: int = int(os.environ.get("AUDIO_SAMPLE_RATE", "44100"))
CHANNELS: int = int(os.environ.get("AUDIO_CHANNELS", "1"))
BUFFER_SECONDS: float = float(os.environ.get("AUDIO_BUFFER_SECONDS", "10"))
DEVICE: str | int | None = os.environ.get("AUDIO_DEVICE") or None

# sounddevice uses int16 frames; numpy dtype to match
DTYPE = "int16"

# ── logging (stderr only — stdout is reserved for native messaging) ────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[audio_capture] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── rolling PCM buffer ────────────────────────────────────────────────────────

# Each entry is a numpy array of shape (n_frames, CHANNELS) with dtype int16.
# Total frames kept = BUFFER_SECONDS * SAMPLE_RATE.
_MAX_FRAMES = int(BUFFER_SECONDS * SAMPLE_RATE)
_buf: deque[np.ndarray] = deque()
_buf_frames: int = 0  # total frames currently held across all chunks
_buf_lock = threading.Lock()


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """Called by sounddevice on each audio block; appends a copy to the buffer."""
    if status:
        logger.warning("sounddevice status: %s", status)

    chunk = indata.copy()  # indata is a view; copy before releasing
    with _buf_lock:
        global _buf_frames
        _buf.append(chunk)
        _buf_frames += frames

        # Trim oldest chunks until we're within the max buffer size
        while _buf_frames > _MAX_FRAMES and _buf:
            oldest = _buf.popleft()
            _buf_frames -= len(oldest)


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


# ── main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info(
        "Starting — rate=%d Hz, channels=%d, buffer=%gs, device=%r",
        SAMPLE_RATE, CHANNELS, BUFFER_SECONDS, DEVICE,
    )

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=DEVICE,
        callback=_audio_callback,
    )

    with stream:
        logger.info("Audio stream open, waiting for messages")
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
                    _write_message({"audio": audio_b64})
                except Exception as exc:
                    logger.exception("Error encoding audio")
                    _write_message({"error": str(exc)})

            elif command == "ping":
                _write_message({"pong": True})

            else:
                logger.warning("Unknown command: %r", command)
                _write_message({"error": f"unknown command: {command!r}"})


if __name__ == "__main__":
    main()
