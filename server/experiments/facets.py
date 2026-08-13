#!/usr/bin/env python3
"""The facet schema for reviewed frames, and a first pass at filling it in.

A ruling in `review_verdicts.json` answers one question — what should the matrix
do — and the review turned up four more that were being forced through the same
field:

| axis | question | why it cannot ride on the ruling |
|---|---|---|
| `video` / `audio` | what is actually on screen, and in the sound? | they disagree, and that disagreement *is* the hard case |
| `where` | where in the break cycle is this? | the same frame means different things at a rejoin and mid-race |
| `care_away` / `care_back` | how much does being wrong cost, in each direction? | the costs are wildly asymmetric and the ruling cannot express that |
| `risk` | is this label safe to *learn* from? | a frame can be correctly `ad` and still be a bad teaching example |

The case that forced this is the post-ad-break ad read: racing on screen, the
network bug visible, and the commentator reading sponsor copy. As a ruling it is
unanswerable, which is why 70 of them landed in `other`. As a coordinate it is
ordinary — `video=live_race, audio=ad_read, where=rejoin` — and it explains why
asking a vision model to catch it never worked: the ad-ness is entirely in the
audio.

`other` was carrying three unrelated things (ad reads, pre-race hype,
transitions) whose desired actions point in opposite directions, which is why no
single treatment of it was right and why the policy re-scoring moved with it.

## Provenance

Nothing here is a ruling. Every value records where it came from:

- `derived` — computed from the segment structure or the OpenCV signals. Trust it.
- `inferred` — pattern-matched from the operator's free-text note. A guess, and
  the note is right there to check it against.
- `default` — the unremarkable case for that ruling and position. Trust it about
  as far as the defaults table below.

Only `inferred` and the frames flagged `review` need a human.

Usage:
    uv run python experiments/facets.py            # report only
    uv run python experiments/facets.py --write    # write review_facets.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "structure"))

VERDICTS = HERE / "review_verdicts.json"
OUT = HERE / "review_facets.json"
S = HERE / "structure"

# ── Vocabularies ──────────────────────────────────────────────────────────────
# Deliberately small. Each value below was written down by the operator during
# the review; nothing is here because it seemed like a category worth having.

VIDEO = {
    "live_race": "full-screen racing, including in-car and onboard",
    "inset": "racing in a box beside something else - interview, replay, studio",
    "side_by_side": "NASCAR NON STOP: race inset, commercial alongside",
    "bumper": "full-screen network transition banner or wipe",
    "spot": "full-screen commercial",
    "studio": "booth, desk, presenters, no racing on screen",
    "pit": "pit road, paddock, garage, driver interview",
    "crowd": "fans, grandstands, aerials of the venue",
    "black": "blank or near-black frame",
}

AUDIO = {
    "commentary": "the race call",
    "ad_read": "talent reading scripted sponsor copy",
    "spot_audio": "a commercial's own audio",
    "chatter": "non-scripted filler - 'welcome back', track talk",
    "music_bed": "bumper or transition music",
    "silence": "no significant audio",
}

# Two axes, not one. Folding the phase into the break cycle costs the break
# position of every pre-race frame - and the pre-race show is 65% commercial, so
# that is exactly where it is worst to lose it.
PHASE = {
    "pre_race": "before the green flag",
    "race": "green flag onwards",
}

# Where in the break cycle. Fully derived from the ruling timeline.
WHERE = {
    "interior": "well inside a content run",
    "pre_break": "the last few seconds before a break starts",
    "onset": "the opening seconds of a break",
    "in_break": "well inside a break",
    "rejoin": "the first seconds after a break ends",
}

RISK = {
    "safe": "fine to train on",
    "fraught": "correct, but likely to generalise badly - see the note",
}

# Seconds either side of a boundary that count as onset / rejoin / pre_break.
# 20 s because that is how long the operator's notes say an ad read runs, and it
# matches the note's measurement of network material opening a break.
EDGE_S = 20.0
DT = 2.0
EDGE = int(EDGE_S / DT)

# The green flag, supplied by the operator rather than derived.
#
# Nothing in the capture marks it reliably. The pre-race *show* ends around 10
# minutes, but driver introductions, the anthem and the pace laps run for
# another eight, and none of that separates from racing by ad load, run length
# or the scoring pylon - `ticker_edges` is noisy enough that a first-sustained
# rule lands on frame 213, which is still the pre-race show. Frame 520 shows lap
# 4 of 350, which puts green about two minutes earlier.
#
# It is one number per broadcast and it decides how ~170 frames are faceted, so
# it is stated here rather than guessed at.
GREEN_FLAG = 475

# ── Note patterns ─────────────────────────────────────────────────────────────
# Matched against the operator's own wording, including the variants actually
# written ("transitional banner", not "transition banner").

NOTE_PATTERNS: list[tuple[str, str, str]] = [
    ("audio", "ad_read", r"ad read|sponsored bit|reading a spons|ad-read"),
    ("audio", "chatter", r"chatter|welcome back|typical chatter|talking about the track"),
    ("video", "bumper", r"transition(al)? banner|full-screen transition|\bwipe\b|\bbumper\b"),
    ("video", "inset", r"\binset\b|picture-in-picture|\bpip\b|squeeze"),
    ("video", "crowd", r"fans in the stands|\bcrowd\b|grandstand"),
    ("video", "live_race", r"in-car|onboard"),
    ("risk", "fraught",
     r"fraught|\bworry\b|confuse the model|over-?correct|prone to treat|"
     r"not sure it'?s (actually )?worth|would make the model"),
]

# Phrases that speak to cost rather than content.
CARE_LOW = re.compile(
    r"not (necessarily )?super important|isn'?t super important|don'?t care|"
    r"not the end of the world|not as important|not mandatory|"
    r"isn'?t as important|not super important", re.I)
CARE_HIGH = re.compile(
    r"more strongly would want|as fast as possible|fuck this|really want|"
    r"meaningful content|actual meaningful", re.I)
ARTIFACT = re.compile(r"youtube tv overlay|player overlay|mouse cursor|on-screen overlay", re.I)
SIGNAL_IDEA = re.compile(
    r"signal to consider|consider using|fingerprint|signal phrase|prime the switcher|"
    r"worth noting|something to consider", re.I)


def infer_from_note(note: str) -> dict:
    out = {}
    if not note:
        return out
    for axis, value, pat in NOTE_PATTERNS:
        if axis not in out and re.search(pat, note, re.I):
            out[axis] = value
    if ARTIFACT.search(note):
        out["artifact"] = True
    if SIGNAL_IDEA.search(note):
        out["signal_idea"] = True
    if CARE_HIGH.search(note):
        out["care_hint"] = "high"
    elif CARE_LOW.search(note):
        out["care_hint"] = "low"
    return out


# ── Derivation from signals and structure ─────────────────────────────────────

def derive_where(y: list[str]) -> list[str]:
    """Position in the break cycle, from the ruling timeline alone."""
    n = len(y)
    starts = [i for i in range(1, n) if y[i] == "ad" and y[i - 1] != "ad"]
    ends = [i for i in range(1, n) if y[i] != "ad" and y[i - 1] == "ad"]
    where = []
    for i in range(n):
        if y[i] == "ad":
            near_start = any(0 <= i - s < EDGE for s in starts)
            where.append("onset" if near_start else "in_break")
        else:
            if any(0 <= i - e < EDGE for e in ends):
                where.append("rejoin")
            elif any(0 <= s - i <= EDGE for s in starts):
                where.append("pre_break")
            else:
                where.append("interior")
    return where


def derive_video(ev: dict, vis: dict, where: str) -> str | None:
    """What the OpenCV signals already settle without anyone reading a note."""
    if ev.get("black") or vis.get("black_frac", 0) > 0.90:
        return "black"
    if ev.get("banner"):
        return "side_by_side"
    if ev.get("bug"):
        return "live_race"
    # Inside a break with no bug and no banner, it is the commercial itself.
    # Outside one the same emptiness is the pylon mode the note found, so it is
    # left blank rather than guessed at.
    if where in ("in_break", "onset"):
        return "spot"
    return None


# Notes are written on one frame but usually describe a run - "this and the next
# 6 frames", "up to <filename>". Reading that range back is what turns 194 notes
# into facets on the stretches they were about.
RANGE_FILE = re.compile(r"(2026-\d{2}-\d{2}T[\d-]+\+00-00)\.jpg")
RANGE_NEXT = re.compile(r"(?:this and the )?next (\d+|few|several) (?:frames|images|captures|clips)", re.I)
WORD_N = {"few": 4, "several": 6}


def note_range(note: str, i: int, pos: dict[str, int], n: int) -> int:
    """The last frame index a note claims to describe, or `i` if it names none."""
    end = i
    for m in RANGE_FILE.finditer(note):
        j = pos.get(m.group(1) + ".jpg")
        if j is not None and j > end:
            end = j
    m = RANGE_NEXT.search(note)
    if m:
        k = m.group(1)
        end = max(end, i + (WORD_N.get(k.lower()) or int(k)))
    return min(end, n - 1)


def defaults_for(verdict: str, phase: str, where: str, audio: str | None) -> tuple[int, int]:
    """(care_away, care_back) before any note overrides.

    The asymmetry is the point. Leaving a commercial up is the thing the operator
    built this to avoid; getting back to the race a few seconds late during a
    rejoin costs almost nothing, and rushing it is what causes the bounce.
    """
    # An ad is an ad in either phase - the pre-race show is two-thirds
    # commercial, and those are the ones the operator built this to escape.
    if verdict == "ad":
        return (3, 0) if where in ("in_break", "onset") else (2, 0)
    if verdict == "content":
        # Pre-race content is "soft content ... not mandatory racing content I'd
        # judge the classifier super harshly for rejecting". Only the *content*
        # side is discounted; the line above keeps pre-race ads urgent.
        if phase == "pre_race":
            return (0, 0)
        if where == "rejoin":
            return (0, 1)  # do not rush back; an ad read may still be running
        return (0, 3)
    # `other`: the binary does not apply. An ad read leans away; hype leans nowhere.
    if audio == "ad_read":
        return (2, 0)
    if phase == "pre_race":
        return (0, 0)
    return (1, 0)


def build() -> tuple[dict, dict]:
    from timeline import load  # noqa: PLC0415

    rows = load()
    store = json.load(open(VERDICTS))["structure"]
    ev = {r["i"]: r for r in map(json.loads, open(S / "evidence.jsonl"))}
    vis = {r["i"]: r for r in map(json.loads, open(S / "visual.jsonl"))}

    verdicts, notes = [], []
    for r in rows:
        rec = store.get(r["filename"]) or {}
        verdicts.append(rec.get("verdict"))
        notes.append(rec.get("note", ""))

    # The break is what was ruled `ad`, and nothing else. An `other` frame is by
    # definition not part of the break - a post-ad-break ad read runs *after* the
    # commercials have stopped, which is the whole reason it is hard. Carrying
    # `other` forward as `ad` would file every ad read under `in_break` and hide
    # exactly the case this schema exists to name.
    timeline = ["ad" if v == "ad" else "content" for v in verdicts]

    where = derive_where(timeline)
    phase = ["pre_race" if i < GREEN_FLAG else "race" for i in range(len(rows))]

    # Spread each note's inference across the run it names, but never past a
    # frame the operator ruled differently - a note about an ad read stops where
    # the ad read stops.
    pos = {r["filename"]: i for i, r in enumerate(rows)}
    spread: dict[int, dict] = {}
    for i, note in enumerate(notes):
        inf = infer_from_note(note)
        if not inf:
            continue
        end = note_range(note, i, pos, len(rows))
        for j in range(i, end + 1):
            if verdicts[j] != verdicts[i]:
                break
            spread.setdefault(j, {}).update(inf)

    facets, stats = {}, Counter()
    for i, r in enumerate(rows):
        v = verdicts[i]
        if v is None:
            continue
        note = notes[i]
        src = {}
        inf = spread.get(i, {})

        vid = derive_video(ev.get(i, {}), vis.get(i, {}), where[i])
        if inf.get("video"):
            vid, src["video"] = inf["video"], "inferred"
        elif vid:
            src["video"] = "derived"
        aud = inf.get("audio")
        if aud:
            src["audio"] = "inferred"
        elif v == "ad" and where[i] in ("in_break", "onset"):
            aud, src["audio"] = "spot_audio", "default"
        elif v == "content" and where[i] in ("interior", "pre_break"):
            aud, src["audio"] = "commentary", "default"

        away, back = defaults_for(v, phase[i], where[i], aud)
        src["care"] = "default"
        if inf.get("care_hint") == "low":
            away, back, src["care"] = min(away, 1), min(back, 1), "inferred"
        elif inf.get("care_hint") == "high":
            away, back = (3, back) if v == "ad" else (away, 3)
            src["care"] = "inferred"

        f = {
            "video": vid,
            "audio": aud,
            "phase": phase[i],
            "where": where[i],
            "care_away": away,
            "care_back": back,
            "risk": inf.get("risk", "safe"),
            "src": src,
        }
        if inf.get("artifact"):
            f["artifact"] = True
        if inf.get("signal_idea"):
            f["signal_idea"] = True
        # Anything guessed from prose, or undecidable with a note explaining why,
        # is worth a human glance. Everything else is derived or unremarkable.
        if "inferred" in src.values() or (v == "other" and note):
            f["review"] = True
        facets[r["filename"]] = f

        stats["frames"] += 1
        stats[f"phase:{phase[i]}"] += 1
        stats[f"where:{where[i]}"] += 1
        stats[f"video:{vid or '-'}"] += 1
        stats[f"audio:{aud or '-'}"] += 1
        stats[f"risk:{f['risk']}"] += 1
        for k, s in src.items():
            stats[f"src:{k}:{s}"] += 1
        if f.get("review"):
            stats["needs review"] += 1
    return facets, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    facets, stats = build()

    print(f"{stats['frames']} frames faceted\n")
    for group in ("phase", "where", "video", "audio", "risk"):
        keys = sorted((k for k in stats if k.startswith(group + ":")),
                      key=lambda k: -stats[k])
        print(f"{group}:")
        for k in keys:
            print(f"    {k.split(':', 1)[1]:14s} {stats[k]:5d}")
    print("\nprovenance:")
    for k in sorted(k for k in stats if k.startswith("src:")):
        print(f"    {k[4:]:22s} {stats[k]:5d}")
    print(f"\nflagged for review: {stats['needs review']}")
    extras = sum(1 for f in facets.values() if f.get("signal_idea"))
    art = sum(1 for f in facets.values() if f.get("artifact"))
    print(f"carrying a signal idea: {extras}   capture artifacts: {art}")

    if args.write:
        with open(OUT, "w") as fh:
            json.dump({"schema": {"video": VIDEO, "audio": AUDIO, "phase": PHASE, "where": WHERE,
                                  "risk": RISK, "care": "0 don't care … 3 urgent"},
                       "facets": facets}, fh, indent=1, sort_keys=True)
        print(f"\nwrote {OUT}")
    else:
        print("\n(dry run — pass --write to save)")


if __name__ == "__main__":
    main()
