"""Score every temporal policy over the labelled race window.

Reports what the operator actually feels:

  accuracy      fraction of frames the matrix is on the right input. With a ~2 s
                cadence this is time in the right state.
  balanced      mean of ad recall and content recall, since content outruns ad
                roughly 3:1 and a policy that never switches would score 73%.
  flaps         emitted switches that are reversed within FLAP_SECS. This is the
                behaviour that made the operator hit the report button - the
                picture bouncing between inputs during one commercial.
  wrong switch  emitted switches into a state that disagrees with ground truth.
  llm%          share of frames that paid for an LLM pass.
  median/p95/max  simulated per-frame wall time.

Uncertain frames stay in the stream - the live system has to do something with
them - but are excluded from accuracy, because a coin-flip label there would be
indistinguishable from classifier error.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from arms import (  # noqa: E402
    BugMemory, Decision, DuplicateCache, Frame, Hysteresis, LogOdds, MajorityK,
    Policy, ProductionDebounce, Score, Stateless,
)

FLAP_SECS = 8.0

# Which capture to score: the burst archive, or the continuous recording.
DATASET, REPLAY, OUT_NAME = "dataset.jsonl", "replay.json", "results.json"
if len(sys.argv) > 1 and sys.argv[1] == "continuous":
    DATASET, REPLAY, OUT_NAME = (
        "cont_dataset.jsonl", "cont_replay.json", "results_continuous.json")
elif len(sys.argv) > 1 and sys.argv[1] == "race":
    # Green flag onward. The operator does not auto-switch during the pre-race
    # show, so the two produced opening montages that dominate the error are
    # out of scope; scoring them would rank policies on frames nobody switches.
    DATASET, REPLAY, OUT_NAME = (
        "race_dataset.jsonl", "cont_replay.json", "results_race.json")


def load(rep: int) -> list[Frame]:
    rows = [json.loads(line) for line in open(HERE / DATASET)]
    replay = json.load(open(HERE / REPLAY))
    t0 = None
    import datetime as dt

    frames = []
    for r in rows:
        t = dt.datetime.fromisoformat(r["timestamp"])
        if t0 is None:
            t0 = t
        runs = replay.get(r["filename"])
        if not runs:
            continue
        run = runs[rep % len(runs)]
        reason = run["reason"]
        source = "opencv" if reason in ("side_by_side", "network_logo", "phash_override") else "llm"
        # Frames the OpenCV pass settles never reach the model, so their
        # measured time is the OpenCV cost, not an LLM cost.
        llm_secs = 0.0 if source == "opencv" else max(0.0, run["secs"] - 0.0024)

        gt = r["gt"]
        # A frame carrying the live upper-right network bug is on-air by direct
        # observation, which settles some of the bumper/billboard grey zone.
        if gt == "uncertain" and (r["usa"] >= 0.65 or r["peacock"] >= 0.55):
            gt = "content"

        frames.append(Frame(
            index=r["i"], episode=r["episode"], seconds=(t - t0).total_seconds(),
            gt=gt, verdict=run["type"], reason=reason, source=source,
            llm_secs=llm_secs, peacock=r["peacock"], usa=r["usa"], sbs=r["sbs"],
            phash_dist_prev=r["phash_dist_prev"],
        ))
    return frames


def initial_states(frames: list[Frame]) -> dict[int, str]:
    """Ground-truth state entering each episode.

    Neutral across policies: no policy carries in its own earlier mistakes and
    none is punished for another's.
    """
    out: dict[int, str] = {}
    last = "content"
    seen: set[int] = set()
    for f in frames:
        if f.episode not in seen:
            out[f.episode] = last
            seen.add(f.episode)
        if f.gt in ("ad", "content"):
            last = f.gt
    return out


def boundary_mask(frames: list[Frame], window: int = 3) -> list[bool]:
    """True for frames within `window` frames after a ground-truth change.

    The archive is biased toward exactly these frames: a burst gets written when
    the classifier changes its mind, so transitions are over-represented by
    roughly 4x against a real broadcast. Lag-based policies pay their whole cost
    here and earn their whole benefit elsewhere, so the two have to be scored
    apart or the bias decides the ranking.
    """
    mask = [False] * len(frames)
    prev_gt = None
    prev_ep = None
    countdown = 0
    for k, f in enumerate(frames):
        if f.episode != prev_ep:
            prev_ep, prev_gt, countdown = f.episode, None, 0
        if f.gt in ("ad", "content"):
            if prev_gt is not None and f.gt != prev_gt:
                countdown = window
            prev_gt = f.gt
        if countdown > 0:
            mask[k] = True
            countdown -= 1
    return mask


def run(policy: Policy, frames: list[Frame], init: dict[int, str]) -> Score:
    s = Score()
    episode = None
    prev_state = None
    switch_times: list[tuple[float, str]] = []
    prev_gt = None
    bmask = boundary_mask(frames)
    steady = [0, 0]     # correct, total
    bound = [0, 0]

    for k, f in enumerate(frames):
        if f.episode != episode:
            episode = f.episode
            policy.reset(init[f.episode])
            prev_state = policy.state
            prev_gt = None

        d: Decision = policy.step(f)
        s.latencies.append(d.secs)
        if d.called_llm:
            s.llm_calls += 1

        if d.state != prev_state:
            s.switches += 1
            switch_times.append((f.seconds, d.state))
            if f.gt in ("ad", "content") and d.state != f.gt:
                s.__dict__.setdefault("wrong_switches", 0)
                s.__dict__["wrong_switches"] = s.__dict__.get("wrong_switches", 0) + 1
        prev_state = d.state

        if f.gt in ("ad", "content"):
            s.frames += 1
            if d.state == f.gt:
                s.correct += 1
            if f.gt == "ad":
                s.ad_total += 1
                s.ad_correct += d.state == "ad"
            else:
                s.content_total += 1
                s.content_correct += d.state == "content"
            if prev_gt is not None and f.gt != prev_gt:
                s.gt_changes += 1
            prev_gt = f.gt
            tgt = bound if bmask[k] else steady
            tgt[1] += 1
            tgt[0] += d.state == f.gt

    s.__dict__["steady"] = steady
    s.__dict__["boundary"] = bound
    flaps = 0
    for i in range(len(switch_times) - 1):
        t1, st1 = switch_times[i]
        t2, st2 = switch_times[i + 1]
        if st2 != st1 and (t2 - t1) <= FLAP_SECS:
            flaps += 1
    s.__dict__["flaps"] = flaps
    return s


def main() -> None:
    reps = len(json.load(open(HERE / REPLAY)).popitem()[1])
    policies_factory = [
        lambda: Stateless(),
        lambda: ProductionDebounce(),
        lambda: MajorityK(3),
        lambda: MajorityK(5),
        lambda: Hysteresis(2, 2),
        lambda: Hysteresis(3, 2),
        lambda: Hysteresis(3, 2, opencv_immediate=True),
        lambda: Hysteresis(4, 2, opencv_immediate=True),
        lambda: Hysteresis(3, 3, opencv_immediate=True),
        lambda: LogOdds(2.0, 1.5, 0.7),
        lambda: LogOdds(2.5, 1.2, 0.8),
        lambda: BugMemory(2, 2),
        lambda: BugMemory(3, 2),
        lambda: DuplicateCache(Hysteresis(3, 2, opencv_immediate=True), 4),
        lambda: DuplicateCache(Hysteresis(3, 2, opencv_immediate=True), 8),
        lambda: DuplicateCache(BugMemory(3, 2), 8),
        lambda: DuplicateCache(Stateless(), 8),
    ]

    header = (f"{'policy':30s} {'acc%':>6s} {'bal%':>6s} {'steady%':>8s} {'bound%':>7s} "
              f"{'adRec%':>7s} {'sw':>4s} {'flap':>5s} {'bad':>4s} {'llm%':>6s} "
              f"{'med':>6s} {'p95':>6s} {'max':>6s}")
    print(header)
    print("-" * len(header))

    results = {}
    for factory in policies_factory:
        accs, bals, adr, conr, sws, flaps, bad, llm = [], [], [], [], [], [], [], []
        std, bnd = [], []
        lat_all: list[float] = []
        name = ""
        for rep in range(reps):
            frames = load(rep)
            init = initial_states(frames)
            p = factory()
            s = run(p, frames, init)
            accs.append(100 * s.accuracy)
            bals.append(100 * s.balanced)
            adr.append(100 * s.ad_correct / s.ad_total)
            conr.append(100 * s.content_correct / s.content_total)
            sws.append(s.switches)
            flaps.append(s.__dict__["flaps"])
            bad.append(s.__dict__.get("wrong_switches", 0))
            llm.append(100 * s.llm_calls / len(frames))
            sc, sn = s.__dict__["steady"]
            bc, bn = s.__dict__["boundary"]
            std.append(100 * sc / sn if sn else 0.0)
            bnd.append(100 * bc / bn if bn else 0.0)
            lat_all.extend(s.latencies)
            name = p.name
        lat_all.sort()
        med = st.median(lat_all)
        p95 = lat_all[int(0.95 * len(lat_all))]
        mx = lat_all[-1]
        print(f"{name:30s} {st.mean(accs):6.1f} {st.mean(bals):6.1f} {st.mean(std):8.1f} "
              f"{st.mean(bnd):7.1f} {st.mean(adr):7.1f} {st.mean(sws):4.0f} "
              f"{st.mean(flaps):5.1f} {st.mean(bad):4.1f} {st.mean(llm):6.1f} "
              f"{med:6.3f} {p95:6.3f} {mx:6.3f}")
        results[name] = {
            "acc": st.mean(accs), "bal": st.mean(bals), "ad_recall": st.mean(adr),
            "content_recall": st.mean(conr), "steady": st.mean(std),
            "boundary": st.mean(bnd), "switches": st.mean(sws),
            "flaps": st.mean(flaps), "wrong_switches": st.mean(bad),
            "llm_pct": st.mean(llm), "median": med, "p95": p95, "max": mx,
            "acc_per_rep": accs,
        }
    json.dump(results, open(HERE / OUT_NAME, "w"), indent=1)

    f0 = load(0)
    gt_changes = run(Stateless(), f0, initial_states(f0)).gt_changes
    print(f"\nframes scored: {sum(1 for f in f0 if f.gt != 'uncertain')}"
          f"  (uncertain excluded: {sum(1 for f in f0 if f.gt == 'uncertain')})"
          f"  ground-truth changes: {gt_changes}  reps: {reps}")


if __name__ == "__main__":
    main()
