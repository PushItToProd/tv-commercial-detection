"""Is the state machine's result an optimum or a coincidence?

The dwell floors were chosen from the same broadcast they are scored on, so the
headline numbers are optimistic by construction. What matters is whether they
sit on a plateau - a wide range of settings that all behave well - or on a spike
that only this broadcast produces. A plateau transfers; a spike does not.

Also reported: the floors are conservative on purpose. `min_ad` is 100 s against
a shortest observed break of 118 s and `min_content` is 80 s against a shortest
observed content run of 94 s, so each has roughly 15% of headroom before it
would start cutting a real segment short.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import score, transitions  # noqa: E402
from ground_truth import build as build_segs  # noqa: E402
from policies import hsmm  # noqa: E402
from timeline import load  # noqa: E402

HERE = Path(__file__).parent
KEY = "p_audio"


def main():
    rows = load()
    segs = build_segs(rows)
    ev = [json.loads(line) for line in open(HERE / "evidence.jsonl")]
    y = []
    for s in segs:
        y.extend([s["label"]] * (s["end"] - s["start"] + 1))
    y = y[: len(ev)]
    n_tr = len(transitions(y))

    print(f"true transitions: {n_tr}\n")
    print("min_ad sweep (min_content=40, cusum=2.0)")
    print(
        f"  {'frames':>7} {'sec':>5} {'acc%':>7} {'sw':>4} {'flap':>5} "
        f"{'ad_shown':>9} {'race_miss':>10}"
    )
    for m in (0, 10, 20, 30, 40, 50, 55, 58, 65, 80):
        r = score(y, hsmm(ev, KEY, min_ad=m))
        print(
            f"  {m:7d} {m * 2:5d} {r['acc'] * 100:7.2f} {r['switches']:4d} "
            f"{r['flaps']:5d} {r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s"
        )

    print("\nmin_content sweep (min_ad=50, cusum=2.0)")
    print(
        f"  {'frames':>7} {'sec':>5} {'acc%':>7} {'sw':>4} {'flap':>5} "
        f"{'ad_shown':>9} {'race_miss':>10}"
    )
    for m in (0, 5, 10, 20, 30, 40, 46, 55, 70):
        r = score(y, hsmm(ev, KEY, min_content=m))
        print(
            f"  {m:7d} {m * 2:5d} {r['acc'] * 100:7.2f} {r['switches']:4d} "
            f"{r['flaps']:5d} {r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s"
        )

    print("\ncusum threshold sweep (min_ad=50, min_content=40)")
    print(
        f"  {'nats':>7} {'acc%':>7} {'sw':>4} {'flap':>5} {'ad_shown':>9} "
        f"{'race_miss':>10}"
    )
    for c in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        r = score(y, hsmm(ev, KEY, cusum_th=c))
        print(
            f"  {c:7.1f} {r['acc'] * 100:7.2f} {r['switches']:4d} {r['flaps']:5d} "
            f"{r['ad_shown']:8.0f}s {r['race_missed']:9.0f}s"
        )

    print("\nnonstop forced length (min_ad=50, min_content=40, cusum=2.0)")
    for n in (0, 30, 55, 60, 65, 90):
        r = score(y, hsmm(ev, KEY, nonstop_len=n))
        print(
            f"  {n:3d} frames  acc {r['acc'] * 100:6.2f}  sw {r['switches']:3d}  "
            f"flap {r['flaps']:2d}  ad_shown {r['ad_shown']:5.0f}s  "
            f"race_miss {r['race_missed']:5.0f}s"
        )


if __name__ == "__main__":
    main()
