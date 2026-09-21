"""Summarize results.jsonl from run.py. "yes" (racing-related) is the content answer."""

import json
import math
from collections import defaultdict
from pathlib import Path

import sys
rows = [json.loads(l) for l in (Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "results.jsonl")).open()]


def auc(pos, neg):
    """P(score of a random content frame > score of a random ad frame)."""
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def conf(p):  # distance from a coin toss, 0..1
    return abs(2 * p - 1)


for mode in ("image_only", "with_audio"):
    rs = [r for r in rows if mode in r]
    print(f"\n===== {mode}  (n={len(rs)}) =====")
    off = [1 - r[mode]["p_yes"] - r[mode]["p_no"] for r in rs]
    print(f"mass off yes/no: max {max(off):.4f}")
    mism = [r["filename"] for r in rs if ("yes" in r[mode]["reply"].lower()) != (r[mode]["p_yes_norm"] > 0.5)]
    print(f"reply disagrees with argmax: {len(mism)}")
    for subset_name, pred in (("all", lambda r: True), ("reaches LLM (opencv undecided)", lambda r: r["opencv"] is None)):
        sub = [r for r in rs if pred(r)]
        pos = [r[mode]["p_yes_norm"] for r in sub if r["verdict"] == "content"]
        neg = [r[mode]["p_yes_norm"] for r in sub if r["verdict"] == "ad"]
        print(f"\n-- {subset_name}: {len(pos)} content, {len(neg)} ad; AUC={auc(pos, neg):.3f}")
        for thr in (0.5, 0.9, 0.99, 0.999):
            tp = sum(p > thr for p in pos); fp = sum(n > thr for n in neg)
            print(f"   reject as ad if p_yes<={thr}: ads caught {len(neg)-fp}/{len(neg)}, content wrongly rejected {len(pos)-tp}/{len(pos)}")
        # wrong answers by confidence
        wrong = [(r, r[mode]["p_yes_norm"]) for r in sub if (r[mode]["p_yes_norm"] > 0.5) != (r["verdict"] == "content")]
        bins = defaultdict(int)
        for r, p in wrong:
            c = conf(p)
            bins["coin toss (<0.6 conf)" if c < 0.6 else "leaning (0.6-0.9)" if c < 0.9 else "confident (>=0.9, p<0.05 or >0.95)"] += 1
        print(f"   wrong: {len(wrong)}  by confidence: {dict(bins)}")
    print("\n-- by stratum (mean p_yes, median confidence, # wrong / n)")
    by = defaultdict(list)
    for r in rs:
        by[(r["verdict"], r["stratum"])].append(r)
    for (v, s), g in sorted(by.items()):
        ps = sorted(r[mode]["p_yes_norm"] for r in g)
        w = sum((p > 0.5) != (v == "content") for p in ps)
        ov = sum(r["opencv"] is None for r in g)
        print(f"   {v:7s} {s:18s} n={len(g):2d} reachLLM={ov:2d} mean_p_yes={sum(ps)/len(ps):.3f} min={ps[0]:.3f} max={ps[-1]:.3f} wrong={w}")
    print("\n-- ambiguous (care_away&care_back>0) / fraught vs rest: mean confidence")
    for label, pred in (("ambiguous", lambda r: r["ambiguous"]), ("fraught", lambda r: r["risk"] == "fraught"),
                        ("noted", lambda r: bool(r["note"])), ("neither", lambda r: not r["ambiguous"] and r["risk"] != "fraught" and not r["note"])):
        for v in ("ad", "content"):
            g = [conf(r[mode]["p_yes_norm"]) for r in rs if pred(r) and r["verdict"] == v]
            if g:
                print(f"   {label:9s} {v:7s} n={len(g):2d} mean conf={sum(g)/len(g):.3f}  conf<0.8: {sum(c<0.8 for c in g)}")

both = [r for r in rows if "with_audio" in r]
flip = [r for r in both if (r["image_only"]["p_yes_norm"] > .5) != (r["with_audio"]["p_yes_norm"] > .5)]
print(f"\nimage vs audio verdict flips: {len(flip)}/{len(both)}")
for r in flip:
    print(f"   {r['verdict']:7s} {r['stratum']:18s} img={r['image_only']['p_yes_norm']:.3f} aud={r['with_audio']['p_yes_norm']:.3f} opencv={r['opencv']}")
print("\nsilent clips:", sum(r["clip_silent"] for r in rows))
print("opencv verdicts:", {k: sum(1 for r in rows if r['opencv']==k) for k in ('ad','content',None)})
print("opencv wrong:", [(r['verdict'], r['stratum'], r['opencv']) for r in rows if r['opencv'] and r['opencv'] != r['verdict']])
