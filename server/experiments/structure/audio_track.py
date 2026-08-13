"""Rebuild a continuous audio track from the overlapping per-frame clips.

Each clip holds the last ~4 s of system audio as of its frame, and frames arrive
every ~2 s, so consecutive clips overlap by about half. Taking the final
`t[i] - t[i-1]` seconds of each clip and concatenating tiles the timeline
exactly once.

Cross-correlating consecutive clips puts the true overlap within ~10 ms of the
nominal frame spacing, so the nominal spacing is used directly - the residual
seam is far below anything speech recognition cares about. The clips are *not*
sample-identical in their overlap (r ~ 0.5), which is expected for a live
stream sampled twice rather than a sign of misalignment.
"""

import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

AUDIO = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/"
    "USA_4K_Iowa_Corn_350/audio"
)
SR = 44100


def read_clip(filename: str) -> np.ndarray:
    p = AUDIO / (Path(filename).stem + ".wav")
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2")


def build(rows, i0: int, i1: int) -> np.ndarray:
    """Continuous PCM covering frames [i0, i1] inclusive."""
    parts = []
    for i in range(max(0, i0), min(len(rows), i1 + 1)):
        clip = read_clip(rows[i]["filename"])
        if i == max(0, i0):
            parts.append(clip)
            continue
        dt = rows[i]["t"] - rows[i - 1]["t"]
        n = min(len(clip), int(round(dt * SR)))
        parts.append(clip[-n:])
    return np.concatenate(parts) if parts else np.zeros(0, "<i2")


def write_wav(path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())
