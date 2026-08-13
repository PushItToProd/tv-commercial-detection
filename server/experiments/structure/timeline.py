"""Shared loading + run-length view of the broadcast.

`load()` joins the visual and audio feature files on filename and returns one
row per frame in capture order, with `t` seconds from the start of the
recording. `runs()` collapses any per-frame series into (value, start, end)
segments, which is how every structural question here gets asked.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

PEACOCK_TH = 0.55
USA_TH = 0.65
SBS_TH = 0.8


def load(visual="visual.jsonl", audio="audio.jsonl"):
    vis = [json.loads(line) for line in open(HERE / visual)]
    aud = {}
    apath = HERE / audio
    if apath.exists():
        aud = {r["filename"]: r for r in (json.loads(line) for line in open(apath))}

    rows = []
    t0 = None
    for r in vis:
        off = r.get("video_offset")
        if t0 is None:
            t0 = off
        a = aud.get(r["filename"], {})
        rows.append(
            {
                **r,
                "t": (off - t0) if off is not None else None,
                "audio": None if "error" in a or not a else a,
                "bug": r["peacock"] >= PEACOCK_TH or r["usa"] >= USA_TH,
                "banner": r["sbs"] >= SBS_TH,
            }
        )
    return rows


def cv_verdict(r):
    """What the production OpenCV pass alone would say. None = falls to the LLM."""
    if r["banner"]:
        return "ad"
    if r["bug"]:
        return "content"
    return None


def runs(rows, key):
    """Collapse `key(row)` into [(value, i0, i1_inclusive), ...]."""
    out = []
    for i, r in enumerate(rows):
        v = key(r)
        if out and out[-1][0] == v:
            out[-1][2] = i
        else:
            out.append([v, i, i])
    return [tuple(x) for x in out]


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"
