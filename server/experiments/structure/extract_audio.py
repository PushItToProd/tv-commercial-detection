"""Per-clip audio signals for a continuous `record_broadcast.py` capture.

Each clip is the last ~4 s of system audio as of its frame's timestamp, so at a
2 s cadence the clips overlap by half and the timeline has no holes. That
matters: video sampled every 2 s misses the black-and-silent joins between
commercial spots, but the audio covers them continuously.

The features are chosen for one specific discrimination - *live motorsport
coverage vs. anything else* - plus a few generic ones for boundary detection:

engine roar
    A pack of stock cars is a loud, broadband, extremely *stationary* noise
    source. `flatness` (geometric/arithmetic mean of the spectrum), `stationarity`
    (mean frame-to-frame spectral correlation within the clip) and the band
    ratios are what separate that from speech and music, which are tonal and
    non-stationary.

commercial loudness
    Spots are mastered flat and loud: high `rms`, low `crest`, low `dyn_range`.
    Live audio breathes.

boundaries
    `silence_frac` and `min_rms` catch the join between two spots; `flux_max`
    catches a hard cut anywhere in the clip.
"""

import argparse
import json
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import signal

TARGET_SR = 22050
NPERSEG = 1024
HOP = 512
EPS = 1e-10

# Log-ish bands (Hz). The 150-800 band is where the engine fundamental and its
# first harmonics live; 60-150 is mostly rumble and music bass.
BANDS = [
    (0, 60),
    (60, 150),
    (150, 400),
    (400, 800),
    (800, 2000),
    (2000, 5000),
    (5000, 11025),
]


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        sr = w.getframerate()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit, got {width * 8}")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return x, sr


def features(path: Path) -> dict:
    try:
        x, sr = read_wav(path)
    except Exception as exc:
        return {"error": str(exc)}
    if x.size == 0:
        return {"error": "empty"}

    peak = float(np.abs(x).max())
    rms_full = float(np.sqrt(np.mean(x * x)))

    # Downsample for the spectral work; 11 kHz of bandwidth is plenty here.
    if sr != TARGET_SR:
        g = np.gcd(sr, TARGET_SR)
        x_ds = signal.resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    else:
        x_ds = x

    f, _, Z = signal.stft(
        x_ds,
        fs=TARGET_SR,
        nperseg=NPERSEG,
        noverlap=NPERSEG - HOP,
        padded=False,
        boundary=None,
    )
    S = np.abs(Z) ** 2  # power, (freq, time)
    if S.shape[1] < 3:
        return {"error": "too short"}

    total = S.sum(axis=0) + EPS
    # Short-time RMS envelope, in true dBFS so the silence threshold means the
    # same thing here as `audio_silence_threshold` does in the app. Taking it
    # from the STFT magnitudes instead would carry the window's scaling and sit
    # ~30 dB low.
    nf = S.shape[1]
    frames = np.lib.stride_tricks.sliding_window_view(x_ds, NPERSEG)[::HOP][:nf]
    env = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    env_db = 20 * np.log10(env + EPS)

    # --- level / dynamics ---
    rms = float(env.mean())
    crest = float(peak / (rms_full + EPS))
    p95, p50, p10 = (float(np.percentile(env_db, q)) for q in (95, 50, 10))
    silence_thresh = 20 * np.log10(0.002)  # ~ -54 dBFS
    silence_frac = float((env_db < silence_thresh).mean())

    # --- spectral shape (energy-weighted over time) ---
    Sn = S / total  # per-frame normalised spectrum
    centroid = float((Sn * f[:, None]).sum(axis=0).mean())
    cum = np.cumsum(Sn, axis=0)
    rolloff = float(f[np.argmax(cum >= 0.85, axis=0)].mean())
    logS = np.log(S + EPS)
    flatness = float(np.exp(logS.mean(axis=0) - np.log(S.mean(axis=0) + EPS)).mean())

    band = {}
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        band[f"b{lo}_{hi}"] = float((S[m].sum(axis=0) / total).mean())

    # --- temporal structure within the clip ---
    # Stationarity: mean correlation between consecutive normalised spectra.
    A, B = Sn[:, :-1], Sn[:, 1:]
    A0, B0 = A - A.mean(axis=0), B - B.mean(axis=0)
    denom = np.sqrt((A0 * A0).sum(axis=0) * (B0 * B0).sum(axis=0)) + EPS
    stationarity = float(((A0 * B0).sum(axis=0) / denom).mean())

    mag = np.sqrt(S)
    flux = np.sqrt((np.diff(mag, axis=1).clip(min=0) ** 2).sum(axis=0))
    flux_n = flux / (mag[:, 1:].sum(axis=0) + EPS)

    zc = float(np.mean(np.abs(np.diff(np.sign(x_ds))) > 0))

    return {
        "peak": peak,
        "rms": rms,
        "rms_db": float(20 * np.log10(rms_full + EPS)),
        "crest": crest,
        "dyn_range": p95 - p10,
        "env_p50_db": p50,
        "env_p10_db": p10,
        "silence_frac": silence_frac,
        "min_rms_db": float(env_db.min()),
        "centroid": centroid,
        "rolloff": rolloff,
        "flatness": flatness,
        "stationarity": stationarity,
        "flux_mean": float(flux_n.mean()),
        "flux_max": float(flux_n.max()),
        "zcr": zc,
        **band,
    }


def _job(args):
    name, path = args
    return name, features(Path(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.dir)
    recs = [
        json.loads(line)
        for line in open(root / "classifications.jsonl")
        if line.strip()
    ]
    recs.sort(key=lambda r: r["timestamp"])
    adir = root / "audio"

    jobs = []
    for r in recs:
        stem = Path(r["filename"]).stem
        p = adir / f"{stem}.wav"
        if p.exists():
            jobs.append((r["filename"], str(p)))

    print(f"{len(jobs)} clips of {len(recs)} frames", file=sys.stderr)
    # Threads, not processes: the work is numpy/scipy FFTs that release the GIL,
    # and a ProcessPoolExecutor needs a socketpair the sandbox won't grant.
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for k, (name, feat) in enumerate(ex.map(_job, jobs)):
            out.append({"filename": name, **feat})
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(out)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
