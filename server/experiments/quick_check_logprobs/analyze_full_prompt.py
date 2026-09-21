"""Summarize results_full_prompt.jsonl from full_prompt.py, beside the quick check."""

import json
import statistics
from collections import defaultdict
from pathlib import Path

rows = [json.loads(line) for line in (Path(__file__).parent / "results_full_prompt.jsonl").open()]
MODES = ("image_only", "with_audio")


def auc(pos, neg):
    """P(a random ad frame scores a higher P(ad) than a random content frame)."""
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def band(p):
    c = abs(2 * p - 1)
    return "confident" if c >= 0.9 else "leaning" if c >= 0.6 else "coin toss"


print(f"relabeled since the quick-check run: {sum(r['verdict'] != r['verdict_before'] for r in rows)}")

print("\n== timing (ms, median / p90) ==")
for m in MODES:
    t = defaultdict(list)
    for r in rows:
        for k in ("wall_ms", "prompt_ms", "predicted_ms", "predicted_n", "prompt_n"):
            t[k].append(r[m][k])
    q = {k: (statistics.median(v), statistics.quantiles(v, n=10)[-1]) for k, v in t.items()}
    print(f"  {m:10s} " + "  ".join(f"{k} {a:.0f}/{b:.0f}" for k, (a, b) in q.items()))

subsets = {
    "stratified, all": lambda r: r["sample"] == "results.jsonl",
    "stratified, OpenCV undecided": lambda r: r["sample"] == "results.jsonl" and r["opencv"] is None,
    "random undecided": lambda r: r["sample"] == "results_undecided.jsonl",
}
for name, pred in subsets.items():
    sub = [r for r in rows if pred(r)]
    ads = [r for r in sub if r["verdict"] == "ad"]
    cons = [r for r in sub if r["verdict"] == "content"]
    print(f"\n== {name}: {len(ads)} ad, {len(cons)} content ==")
    for m in MODES:
        wrong = [r for r in sub if (r[m]["p_ad"] > 0.5) != (r["verdict"] == "ad")]
        missed = sum(r[m]["p_ad"] <= 0.5 for r in ads)
        false_ad = sum(r[m]["p_ad"] > 0.5 for r in cons)
        bands = defaultdict(int)
        for r in wrong:
            bands[band(r[m]["p_ad"])] += 1
        print(f"  full {m:10s} ads passed as content {missed}/{len(ads)}, content called ad {false_ad}/{len(cons)},"
              f" AUC {auc([r[m]['p_ad'] for r in ads], [r[m]['p_ad'] for r in cons]):.3f}, wrong by band {dict(bands)}")
    # production cascade: quick check (audio prompt, reply at 0.5) then full prompt with audio
    for qk, fm, label in (("quick_audio_p_yes", "with_audio", "production (audio)"),
                          ("quick_image_p_yes", "image_only", "image-only")):
        def call(r):
            return "ad" if r[qk] <= 0.5 or r[fm]["p_ad"] > 0.5 else "content"
        missed = sum(call(r) == "content" for r in ads)
        false_ad = sum(call(r) == "ad" for r in cons)
        print(f"  cascade {label:18s} ads passed as content {missed}/{len(ads)}, content called ad {false_ad}/{len(cons)}")

print("\n== by stratum: n, mean P(ad) img / aud, wrong img / aud ==")
by = defaultdict(list)
for r in rows:
    by[(r["sample"][:12], r["verdict"], r["stratum"])].append(r)
for (s, v, st), g in sorted(by.items()):
    cells = []
    for m in MODES:
        ps = [r[m]["p_ad"] for r in g]
        w = sum((p > 0.5) != (v == "ad") for p in ps)
        cells.append(f"{sum(ps) / len(ps):.2f} w{w}")
    print(f"  {s:12s} {v:7s} {st:18s} n={len(g):2d}  img {cells[0]:9s} aud {cells[1]}")

print("\n== confidence (|2p-1|) by operator flag, stratified sample ==")
for label, pred in (("ambiguous", lambda r: r["ambiguous"]), ("fraught", lambda r: r["risk"] == "fraught"),
                    ("unflagged", lambda r: not r["ambiguous"] and r["risk"] != "fraught" and not r["note"])):
    for v in ("ad", "content"):
        g = [r for r in rows if r["sample"] == "results.jsonl" and pred(r) and r["verdict"] == v]
        if g:
            cells = [f"{statistics.mean(abs(2 * r[m]['p_ad'] - 1) for r in g):.3f}" for m in MODES]
            print(f"  {label:9s} {v:7s} n={len(g):2d} img {cells[0]} aud {cells[1]}")

print("\n== all wrong answers ==")
for r in rows:
    for m in MODES:
        if (r[m]["p_ad"] > 0.5) != (r["verdict"] == "ad"):
            print(f"  {m[:5]} {r['verdict']:7s} {r['stratum']:18s} p_ad={r[m]['p_ad']:.3f} ocv={r['opencv']}"
                  f" qa={r['quick_audio_p_yes']:.2f} | {r[m]['reply'].strip()[:110]!r}")
