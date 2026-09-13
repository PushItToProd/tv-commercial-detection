"""Cheap DSP features over the audio clip that arrives with each frame.

Lifted from `experiments/structure/extract_audio.py`, which is what the audio
models were fit on. The arithmetic must stay identical to it — a shipped model's
weights mean nothing against a feature vector computed even slightly
differently — and `tests/test_audio_features.py` checks that it does. Change
both or neither.

The features are chosen for one discrimination, *live motorsport coverage vs.
anything else*, plus a few generic ones for boundaries:

engine roar
    A pack of stock cars is a loud, broadband, extremely *stationary* noise
    source. `flatness`, `stationarity` and the band ratios separate that from
    speech and music, which are tonal and non-stationary.

commercial loudness
    Spots are mastered flat and loud: high `rms`, low `crest`, low `dyn_range`.
    Live audio breathes.

boundaries
    `silence_frac` and `min_rms` catch the join between two spots; `flux_max`
    catches a hard cut anywhere in the clip.
"""

import io
import wave

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

# The order a model's weights are laid out in. `peak` and `rms` are computed but
# not modeled: `peak` is a capture-level artifact and `rms` duplicates `rms_db`.
FEATURES = [
    "rms_db",
    "crest",
    "dyn_range",
    "env_p50_db",
    "env_p10_db",
    "silence_frac",
    "min_rms_db",
    "centroid",
    "rolloff",
    "flatness",
    "stationarity",
    "flux_mean",
    "flux_max",
    "zcr",
    "b0_60",
    "b60_150",
    "b150_400",
    "b400_800",
    "b800_2000",
    "b2000_5000",
    "b5000_11025",
]


def read_wav(wav_bytes: bytes) -> tuple[np.ndarray, int] | None:
    """Samples as float32 in [-1, 1) and the sample rate, or None if unreadable.

    Only 16-bit PCM is accepted, which is what the native host writes. Multiple
    channels are averaged down to one; the models were fit on mono clips.
    """
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            sr = w.getframerate()
            width = w.getsampwidth()
            channels = w.getnchannels()
            raw = w.readframes(w.getnframes())
    except wave.Error, EOFError, ValueError:
        return None
    if width != 2:
        return None
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        x = x[: x.size - x.size % channels].reshape(-1, channels).mean(axis=1)
    return x, sr


def features(wav_bytes: bytes) -> dict[str, float] | None:
    """Every feature for one clip, or None if the clip can't be analyzed."""
    decoded = read_wav(wav_bytes)
    if decoded is None:
        return None
    x, sr = decoded
    if x.size == 0:
        return None

    peak = float(np.abs(x).max())
    rms_full = float(np.sqrt(np.mean(x * x)))

    # Downsample for the spectral work; 11 kHz of bandwidth is plenty here.
    if sr != TARGET_SR:
        g = np.gcd(sr, TARGET_SR)
        x_ds = signal.resample_poly(x, TARGET_SR // g, sr // g).astype(np.float32)
    else:
        x_ds = x

    if x_ds.size < NPERSEG:
        return None
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
        return None

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


def feature_vector(wav_bytes: bytes) -> np.ndarray | None:
    """The modeled features for one clip, in `FEATURES` order."""
    feats = features(wav_bytes)
    if feats is None:
        return None
    return np.array([feats[name] for name in FEATURES], dtype=np.float64)


def trailing_block(history: np.ndarray) -> np.ndarray:
    """The model input for the newest clip in *history* (oldest row first).

    Where we are, and how that differs from recently: the trailing mean and sd
    over the window say what the last ~30 s sounded like, which is what a break
    *is*; the current clip and its difference from the mean are the change
    detector, a step visible on a break's first clip even though the average has
    barely moved. Without them the sensor is slow off the mark at every onset.
    """
    mean = history.mean(axis=0)
    current = history[-1]
    return np.concatenate([mean, history.std(axis=0), current, current - mean])
