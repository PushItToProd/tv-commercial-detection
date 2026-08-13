"""What would a verbal pre-break cue actually be worth?

The state machine's remaining error is almost entirely at full-break onsets: NON
STOP breaks switch in 2 s because the banner is unambiguous, full breaks take a
median of 10 s and up to 36 s, and that lag is the whole of the `ad_shown`
number. Every visual and audio signal in this experiment fires at best on the
break's first frame, so none of them can shorten it.

A commentator saying "we'll be right back" fires *before* the first frame. That
is the only lever on onset latency found anywhere here.

This models it as an arming signal rather than a verdict. A cue does not switch
anything by itself - it lowers the evidence threshold for the next few frames,
so the sensor's first hint of a commercial is acted on immediately instead of
being accumulated. That shape is deliberate: with precision measured at 1.0 and
recall well under 1.0, a cue can only ever pull a switch earlier, and a false
cue costs nothing unless the audio agrees with it.

Honesty note: `CUE_PATTERNS` in transcript_cues.py was written from general US
sports-broadcast convention before the transcript existed, not read off this
broadcast. The control windows are what keep that claim checkable - a phrase
only counts if it is absent from ordinary commentary.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score, transitions  # noqa: E402
from ground_truth import build as build_segs  # noqa: E402
from policies import (  # noqa: E402
    CLAMP,
    MIN_AD_DWELL,
    MIN_CONTENT_DWELL,
    _logodds,
    hsmm,
)
from timeline import load  # noqa: E402
from transcript_cues import CUE_PATTERNS, load_tx  # noqa: E402

HERE = Path(__file__).parent
CUE_WINDOW = 30.0  # seconds a cue stays armed
ARMED_TH = 0.5  # CUSUM threshold while armed, vs 2.0 normally


def cue_times(tx):
    rx = [re.compile(p) for p, n in CUE_PATTERNS if "control" not in n]
    return [s["s"] for s in tx if any(r.search(s["t"].lower()) for r in rx)]


def mark_cues(ev, cues):
    for e in ev:
        e["cue"] = any(0 <= e["t"] - c <= CUE_WINDOW for c in cues)
    return ev


def hsmm_cue(
    ev,
    key,
    min_ad=MIN_AD_DWELL,
    min_content=MIN_CONTENT_DWELL,
    cusum_th=2.0,
    armed_th=ARMED_TH,
):
    out = []
    state, kind, dwell, acc = "content", None, 10**6, 0.0
    for e in ev:
        dwell += 1
        if state == "ad" and kind == "nonstop":
            if dwell >= 60 and not e["banner"]:
                state, kind, dwell, acc = "content", None, 0, 0.0
            out.append(state)
            continue
        if e["banner"] and state != "ad":
            state, kind, dwell, acc = "ad", "nonstop", 0, 0.0
            out.append(state)
            continue
        if e["black"] and state == "content" and dwell >= min_content:
            state, kind, dwell, acc = "ad", "full", 0, 0.0
            out.append(state)
            continue
        floor = min_ad if state == "ad" else min_content
        if dwell < floor:
            acc = 0.0
            out.append(state)
            continue
        if e["anchor"] is not None:
            ll = 8.0 if e["anchor"] == "ad" else -8.0
        elif key is None:
            ll = -1.0
        else:
            ll = _logodds(e[key])
        ll = max(-CLAMP, min(CLAMP, ll))
        acc = max(0.0, acc + (ll if state == "content" else -ll))
        # A cue only relaxes the threshold for *entering* a break.
        th = armed_th if (e.get("cue") and state == "content") else cusum_th
        if acc >= th:
            state = "ad" if state == "content" else "content"
            kind = "full" if state == "ad" else None
            dwell, acc = 0, 0.0
        out.append(state)
    return out


def main():
    rows = load()
    segs = build_segs(rows)
    ev = [json.loads(line) for line in open(HERE / "evidence.jsonl")]
    tx = load_tx()
    horizon = tx[-1]["e"]
    cues = cue_times(tx)
    ev = mark_cues(ev, cues)

    y = []
    for s in segs:
        y.extend([s["label"]] * (s["end"] - s["start"] + 1))
    y = y[: len(ev)]

    # Only score the part of the timeline the transcript actually covers.
    n = sum(1 for e in ev if e["t"] <= horizon)
    print(f"transcript covers {n} of {len(ev)} frames; {len(cues)} cue utterances\n")
    ev_c, y_c = ev[:n], y[:n]

    print(
        f"{'policy':22s} {'acc%':>7} {'sw':>4} {'flap':>5} {'ad_shown':>9} "
        f"{'race_miss':>10} {'onset lag':>10}"
    )
    for name, pred in (
        ("state machine", hsmm(ev_c, "p_audio")),
        ("+ verbal cue", hsmm_cue(ev_c, "p_audio")),
    ):
        r = score(y_c, pred)
        tr = [i for i in transitions(y_c) if y_c[i] == "ad"]
        lags = []
        for t in tr:
            k = next(
                (k for k in range(t, min(len(y_c), t + 60)) if pred[k] == "ad"), None
            )
            lags.append((k - t) * 2 if k is not None else 120)
        print(
            f"{name:22s} {r['acc'] * 100:7.2f} {r['switches']:4d} {r['flaps']:5d} "
            f"{r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s "
            f"{sum(lags) / len(lags):9.1f}s"
        )

    print("\nper-break onset lag (s):")
    base = hsmm(ev_c, "p_audio")
    cued = hsmm_cue(ev_c, "p_audio")
    for t in [i for i in transitions(y_c) if y_c[i] == "ad"]:

        def lag(p):
            k = next((k for k in range(t, min(len(y_c), t + 60)) if p[k] == "ad"), None)
            return (k - t) * 2 if k is not None else 120

        kind = next((s["kind"] for s in segs if s["start"] == t), "")
        armed = any(ev_c[j].get("cue") for j in range(max(0, t - 15), t + 1))
        print(
            f"  frame {t:5d} {kind:8s} base {lag(base):3d}s -> cued {lag(cued):3d}s"
            f"{'   (cue armed)' if armed else ''}"
        )


if __name__ == "__main__":
    main()
