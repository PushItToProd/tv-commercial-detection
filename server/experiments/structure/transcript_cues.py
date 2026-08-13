"""Do the commentators announce the break before it happens?

This is the operator's hypothesis, and it is the only signal examined anywhere in
this experiment that could be *leading* rather than simultaneous. Every visual
check fires at best on the break's first frame; a verbal tease fires seconds
earlier, which is the difference between switching late and switching on time.

Method. For every break, take the transcript of the 40 s before it starts. For a
control, take a 40 s window drawn from well inside a segment - at least 40 s from
either edge - so that a phrase only counts as a cue if it is *absent* from
ordinary commentary. Without the controls this analysis would "discover" that
commentators say "the" before ad breaks.

Then score candidate cue phrases by how many breaks they precede against how
often they fire anywhere else, and report the two rates separately. A cue that
fires before 8 of 17 breaks and never otherwise is useful even though it misses
half of them: it can only ever make the switch earlier, never wrong.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import build as build_segs  # noqa: E402
from timeline import fmt, load  # noqa: E402

HERE = Path(__file__).parent
PRE = 40.0  # seconds before a break to search for a tease
POST = 40.0  # seconds after a return, for the ad read
CTRL_LEN = 40.0

# Phrases a US sports broadcast uses to hand off to a break. Written from the
# general form, not read off this transcript, so that hit rates mean something.
CUE_PATTERNS = [
    (r"\bwe'?ll be (right )?back\b", "we'll be back"),
    (r"\bwhen we come back\b", "when we come back"),
    (r"\bcoming up (next|after)\b", "coming up next"),
    (r"\bis next\b|\bnext here on\b|\bup next\b", "next / next here on"),
    (r"\bstay (with us|tuned)\b", "stay with us"),
    (r"\bside[- ]by[- ]side\b", "side-by-side"),
    (r"\bnascar non ?stop\b", "NASCAR NON STOP"),
    (
        r"\byou won'?t miss\b|\bwithout missing\b|\bnot miss a (lap|thing|beat)\b",
        "you won't miss",
    ),
    (r"\bback (in|after) (a|this) (moment|minute|break)\b", "back after this"),
    (r"\b(quick|short) break\b|\bcommercial break\b", "a break"),
    (r"\bbrought to you by\b|\bpresented by\b", "brought to you by"),
    (r"\bgreen ?flag\b", "green flag (control phrase)"),
]


def load_tx():
    p = HERE / "transcript_full.jsonl"
    return [json.loads(line) for line in open(p) if line.strip()]


def text_in(tx, t0, t1):
    return " ".join(s["t"] for s in tx if s["e"] > t0 and s["s"] < t1).lower()


def main():
    rows = load()
    segs = build_segs(rows)
    tx = load_tx()
    horizon = tx[-1]["e"]
    print(f"transcript covers 0 - {fmt(horizon)} of {fmt(rows[-1]['t'])}\n")

    breaks = [s for s in segs if s["label"] == "ad" and s["t0"] <= horizon]
    returns = [
        s
        for s in segs
        if s["label"] == "content" and s["t0"] <= horizon and s["start"] > 0
    ]
    print(f"{len(breaks)} breaks and {len(returns)} returns inside the transcript\n")

    pre_texts = [(s, text_in(tx, s["t0"] - PRE, s["t0"])) for s in breaks]
    post_texts = [(s, text_in(tx, s["t0"], s["t0"] + POST)) for s in returns]

    rnd = random.Random(1)
    ctrl = []
    for s in segs:
        if s["t1"] - s["t0"] < 3 * CTRL_LEN or s["t0"] > horizon:
            continue
        for _ in range(3):
            a = rnd.uniform(s["t0"] + CTRL_LEN, s["t1"] - 2 * CTRL_LEN)
            ctrl.append((s["label"], text_in(tx, a, a + CTRL_LEN)))
    print(
        f"{len(ctrl)} control windows "
        f"({sum(1 for lab, _ in ctrl if lab == 'ad')} inside breaks)\n"
    )

    print(f"{'cue':28s} {'pre-break':>10} {'control':>9} {'ctrl/ad':>9}")
    print("-" * 60)
    for pat, name in CUE_PATTERNS:
        rx = re.compile(pat)
        hit = sum(1 for _, t in pre_texts if rx.search(t))
        c_all = sum(1 for _, t in ctrl if rx.search(t))
        c_ad = sum(1 for lab, t in ctrl if lab == "ad" and rx.search(t))
        print(
            f"{name:28s} {hit:4d}/{len(pre_texts):<5} {c_all:4d}/{len(ctrl):<4} "
            f"{c_ad:4d}/{sum(1 for lab, _ in ctrl if lab == 'ad'):<4}"
        )

    print("\n--- any cue at all, per break ---")
    anyrx = [re.compile(p) for p, n in CUE_PATTERNS if "control" not in n]
    for s, t in pre_texts:
        got = [
            n
            for (p, n), rx in zip(CUE_PATTERNS, anyrx)
            if "control" not in n and rx.search(t)
        ]
        print(f"  {fmt(s['t0'])} {s['kind']:8s} {', '.join(got) if got else '-'}")

    print("\n--- the last thing said before each break ---")
    for s, _ in pre_texts:
        last = [x for x in tx if x["s"] < s["t0"]][-1:]
        for x in last:
            print(f"  {fmt(s['t0'])} [{x['s'] - s['t0']:+6.1f}s] {x['t']}")

    print("\n--- the first thing said after each return (the ad read) ---")
    for s, t in post_texts:
        first = [x for x in tx if x["s"] >= s["t0"]][:2]
        for x in first:
            print(f"  {fmt(s['t0'])} [{x['s'] - s['t0']:+6.1f}s] {x['t']}")

    print("\n--- words enriched in the 40 s before a break ---")

    def words(ts):
        c = Counter()
        for t in ts:
            c.update(re.findall(r"[a-z']+", t))
        return c

    pre_c = words([t for _, t in pre_texts])
    ctl_c = words([t for _, t in ctrl])
    n_pre, n_ctl = sum(pre_c.values()), sum(ctl_c.values())
    scored = []
    for w, k in pre_c.items():
        if k < 3 or len(w) < 3:
            continue
        r = (k / n_pre) / ((ctl_c.get(w, 0) + 1) / n_ctl)
        scored.append((r, k, ctl_c.get(w, 0), w))
    for r, k, c, w in sorted(scored, reverse=True)[:20]:
        print(f"  {w:16s} x{r:5.1f}   pre {k:3d}  control {c:3d}")


if __name__ == "__main__":
    main()
