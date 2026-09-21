"""Summarize survey_raw.jsonl against survey_sample.jsonl (see survey.py).

"census" is every frame the production cascade sends to the LLM, weight 1.
"audio-decided" is the weighted sample of frames the audio sensor decides.
Production conditions: the audio variant of each pass when the clip isn't
silent, the image-only one when it is.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
BASE = Path("/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com")
MODEL = json.loads(
    (HERE.parent.parent / "src/tv_commercial_detector/classifiers/audio_models/nascar_on_nbc.json").read_text()
)

sample = {(r["broadcast"], r["filename"]): r for r in map(json.loads, (HERE / "survey_sample.jsonl").open())}
raw = defaultdict(dict)
for r in map(json.loads, (HERE / "survey_raw.jsonl").open()):
    raw[(r["broadcast"], r["filename"])][r["condition"]] = r
rows = []
for key, s in sample.items():
    a = raw.get(key, {})
    if "quick_image" not in a or "full_image" not in a:
        continue
    q = a.get("quick_audio", a["quick_image"])
    f = a.get("full_audio", a["full_image"])
    rows.append({**s, **{c: a.get(c) for c in ("quick_image", "quick_audio", "full_image", "full_audio")},
                 "quick": q, "full": f, "silent": "quick_audio" not in a})
print(f"{len(rows)} of {len(sample)} frames answered; {sum(r['silent'] for r in rows)} with silent clips")


def short(b):
    return b.split("_")[2]


def rate(sub, pred):
    w = sum(r["weight"] for r in sub)
    return sum(r["weight"] for r in sub if pred(r)) / w if w else float("nan")


def cell(sub, pred):
    n = sum(pred(r) for r in sub)
    return f"{n}/{len(sub)} ({rate(sub, pred):.0%} wtd)" if any(r["weight"] != 1 for r in sub) else f"{n}/{len(sub)}"


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


# ---- timing ----
print("\n== timing, ms (median / p90) ==")
for c in ("quick_image", "quick_audio", "full_image", "full_audio"):
    rs = [r[c] for r in rows if r[c]]
    fields = ("wall_ms", "prompt_ms", "predicted_ms", "predicted_n")
    print(f"  {c:11s} n={len(rs):3d} " + "  ".join(
        f"{k} {statistics.median(x[k] for x in rs):.0f}/{statistics.quantiles([x[k] for x in rs], n=10)[-1]:.0f}"
        for k in fields))

# ---- population: where frames fall in the cascade ----
print("\n== cascade over the whole broadcast (non-excluded frames) ==")
for b in ("USA_4K_Cook_Out_400", "USA_4K_Iowa_Corn_350"):
    frames = json.loads((BASE / b / "annotations.json").read_text())["frames"]
    tally = defaultdict(int)
    for line in (HERE / f"cascade_{b}.jsonl").open():
        c = json.loads(line)
        ann = frames.get(c["filename"], {})
        if ann.get("verdict") not in ("ad", "content") or ann.get("exclude"):
            continue
        v = ann["verdict"]
        if c["anchor"] is not None:
            tally[("opencv", c["anchor"], v)] += 1
        elif c["p_audio"] is not None and c["p_audio"] >= MODEL["ad_threshold"]:
            tally[("audio", "ad", v)] += 1
        elif c["p_audio"] is not None and c["p_audio"] <= MODEL["content_threshold"]:
            tally[("audio", "content", v)] += 1
        else:
            tally[("llm", "-", v)] += 1
    total = sum(tally.values())
    print(f"  {short(b)} ({total} frames)")
    for st in ("opencv", "audio"):
        for call in ("ad", "content"):
            right, wrong = tally[(st, call, call)], tally[(st, call, "content" if call == "ad" else "ad")]
            if right + wrong:
                print(f"    {st:6s} calls {call:7s} {right + wrong:5d}  wrong {wrong:4d} ({wrong / (right + wrong):.1%})")
    llm = tally[("llm", "-", "ad")] + tally[("llm", "-", "content")]
    print(f"    to the LLM         {llm:5d}  ({llm / total:.1%} of frames; {tally[('llm', '-', 'ad')]} ad, "
          f"{tally[('llm', '-', 'content')]} content)")

# ---- accuracy ----
def is_ad_quick(r, c, thr=0.5):
    return r[c]["p_yes"] <= thr


def is_ad_full(r, c):
    # A reply with no verdict parses as `unknown`, which never switches to ad.
    return r[c]["p_ad"] is not None and r[c]["p_ad"] > 0.5


def p_ad(r, c):
    return 0.5 if r[c]["p_ad"] is None else r[c]["p_ad"]


def report(title, sub):
    ads = [r for r in sub if r["verdict"] == "ad"]
    cons = [r for r in sub if r["verdict"] == "content"]
    print(f"\n== {title}: {len(ads)} ad, {len(cons)} content ==")
    print(f"  {'':34s} {'ads passed as content':24s} {'content called ad':24s} AUC")
    for label, c, fn in (("quick image-only", "quick_image", is_ad_quick), ("quick audio (production)", "quick", is_ad_quick),
                         ("full image-only", "full_image", is_ad_full), ("full audio (production)", "full", is_ad_full)):
        score = (lambda r, c=c: 1 - r[c]["p_yes"]) if c.startswith("quick") else (lambda r, c=c: p_ad(r, c))
        print(f"  {label:34s} {cell(ads, lambda r, c=c, fn=fn: not fn(r, c)):24s} "
              f"{cell(cons, lambda r, c=c, fn=fn: fn(r, c)):24s} {auc([score(r) for r in ads], [score(r) for r in cons]):.3f}")

    def cascade(r):
        return is_ad_quick(r, "quick") or is_ad_full(r, "full")
    print(f"  {'cascade quick -> full (production)':34s} {cell(ads, lambda r: not cascade(r)):24s} {cell(cons, cascade):24s}")
    for thr in (0.1, 0.5, 0.9):
        def cas(r, thr=thr):
            return is_ad_quick(r, "quick", thr) or is_ad_full(r, "full")
        print(f"  {f'cascade, quick rejects at P(yes)<={thr}':34s} {cell(ads, lambda r: not cas(r)):24s} {cell(cons, cas):24s}")

    t = defaultdict(float)
    for r in sub:
        t[(r["verdict"], "no" if is_ad_quick(r, "quick") else "yes", "ad" if is_ad_full(r, "full") else "content")] += 1
    print("  quick check x full prompt (production conditions):")
    for k in sorted(t):
        print(f"    operator {k[0]:7s} quick {k[1]:3s} full {k[2]:7s} {int(t[k])}")
    rejected = sum(is_ad_quick(r, "quick") for r in sub)
    q_ms = statistics.median(r["quick"]["wall_ms"] for r in sub)
    f_ms = statistics.median(r["full"]["wall_ms"] for r in sub)
    print(f"  quick rejects {rejected}/{len(sub)} ({rejected / len(sub):.0%}); break-even {q_ms / f_ms:.0%}"
          f" (quick {q_ms:.0f} ms, full {f_ms:.0f} ms)")
    print(f"  mean LLM time per frame: with quick check {(len(sub) * q_ms + (len(sub) - rejected) * f_ms) / len(sub):.0f} ms,"
          f" full prompt alone {f_ms:.0f} ms")


census = [r for r in rows if r["stage"] == "llm"]
audio = [r for r in rows if r["stage"] == "audio"]
report("census: frames the cascade sends to the LLM, both broadcasts", census)
for b in ("USA_4K_Cook_Out_400", "USA_4K_Iowa_Corn_350"):
    report(f"census, {short(b)}", [r for r in census if r["broadcast"] == b])
report("audio-decided sample (weighted)", audio)

# ---- confidence ----
print("\n== full prompt (production) confidence ==")
for title, sub in (("census", census), ("audio-decided", audio)):
    unknown = sum(r["full"]["p_ad"] is None for r in sub)
    mid = sum(0.05 < p_ad(r, "full") < 0.95 for r in sub) - unknown
    wrong = [r for r in sub if is_ad_full(r, "full") != (r["verdict"] == "ad")]
    conf_wrong = sum(abs(2 * p_ad(r, "full") - 1) >= 0.9 for r in wrong)
    print(f"  {title:13s} P(ad) in (0.05, 0.95): {mid}/{len(sub)}; no verdict: {unknown};"
          f" wrong {len(wrong)}, of which confident {conf_wrong}")

# ---- errors by operator facet / note, census + audio sample ----
print("\n== full prompt (production) errors by operator group, all 500 ==")
g = defaultdict(lambda: [0, 0])
for r in rows:
    key = (r["verdict"], r["video"] or "-", r["group"][:70])
    g[key][1] += 1
    if is_ad_full(r, "full") != (r["verdict"] == "ad"):
        g[key][0] += 1
for (v, video, grp), (w, n) in sorted(g.items(), key=lambda kv: -kv[1][0]):
    if w:
        print(f"  {v:7s} video={video:12s} {w:3d}/{n:<3d} {grp}")

print("\n== full prompt (production) wrong answers, census ==")
for r in sorted(census, key=lambda r: (r["broadcast"], r["filename"])):
    if is_ad_full(r, "full") != (r["verdict"] == "ad"):
        print(f"  {short(r['broadcast']):4s} {r['filename'][11:19]} {r['verdict']:7s} p_ad={p_ad(r, 'full'):.3f}"
              f" q={r['quick']['p_yes']:.2f} {r['full']['reply'].strip()[:120]!r}")
