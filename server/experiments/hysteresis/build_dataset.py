"""Assemble the labelled temporal dataset for the race window.

Combines the per-frame signal table with burst-level and frame-level ground
truth, and splits the sequence into episodes: runs of consecutive frames no more
than `MAX_GAP` seconds apart. An episode is a stretch of real production frames
at the live ~2 s cadence, which is what a temporal policy actually sees; the
archive only keeps such stretches in bursts, so nothing longer can be simulated
honestly.
"""
import json
import sys
from pathlib import Path

import datetime as dt

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import burst_labels  # noqa: E402
import frame_labels  # noqa: E402

MAX_GAP = 10.0


def main() -> None:
    rows = [json.loads(line) for line in open(HERE / "frames_0809.jsonl")]
    by_i = {r["i"]: r for r in rows}
    race = json.load(open(HERE / "race_bursts.json"))

    out = []
    for b, burst in enumerate(race):
        burst_label = burst_labels.label_for(b)
        for i in burst:
            r = dict(by_i[i])
            r["burst"] = b
            r["gt"] = frame_labels.apply(i) or burst_label
            out.append(r)
    out.sort(key=lambda r: r["i"])

    # Episodes: consecutive frames within MAX_GAP seconds.
    episodes: list[list[int]] = []
    cur: list[int] = []
    prev_t = None
    for k, r in enumerate(out):
        t = dt.datetime.fromisoformat(r["timestamp"])
        if prev_t is not None and (t - prev_t).total_seconds() > MAX_GAP:
            episodes.append(cur)
            cur = []
        cur.append(k)
        prev_t = t
    if cur:
        episodes.append(cur)

    for e, ep in enumerate(episodes):
        for k in ep:
            out[k]["episode"] = e

    with open(HERE / "dataset.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    json.dump(episodes, open(HERE / "episodes.json", "w"))

    from collections import Counter
    c = Counter(r["gt"] for r in out)
    print(f"{len(out)} frames, {len(episodes)} episodes")
    print("ground truth:", dict(c))
    lens = sorted(len(e) for e in episodes)
    print(f"episode length: median {lens[len(lens)//2]} max {lens[-1]}"
          f"  episodes>=10 frames: {sum(1 for x in lens if x >= 10)}")

    # Transitions in ground truth, ignoring uncertain frames.
    trans = 0
    prev = None
    for r in out:
        if r["gt"] == "uncertain":
            continue
        if prev is not None and r["gt"] != prev:
            trans += 1
        prev = r["gt"]
    print(f"ground-truth label changes across the window: {trans}")

    # Do the uncertain frames carry the live upper-right USA bug? If they do,
    # the broadcast is on air and the frame is objectively content.
    unc = [r for r in out if r["gt"] == "uncertain"]
    hit = sum(1 for r in unc if r["usa"] >= 0.65 or r["peacock"] >= 0.55)
    print(f"uncertain frames: {len(unc)}, of which {hit} carry a network bug")


if __name__ == "__main__":
    main()
