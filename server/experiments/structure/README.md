# Broadcast structure experiment

Scratch code behind `notes/broadcast-structure-2026-08.md`. Not wired into the
server and not needed to run it — kept so the numbers in that note can be
re-derived or re-run against a new broadcast.

Everything writes into this directory. Unlike the `hysteresis/` experiment,
**nothing here needs llama.cpp**: the whole point was to find out how far the
pipeline gets on cheap local signals alone.

## Dataset

One capture — `record_broadcast.py` output for the 2026-08-13 Iowa Corn 350 on
USA, at `/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Iowa_Corn_350`.
4775 frames over 2:39:35 at a 2.01 s cadence, no gaps, an audio clip for every
frame. Three times the length of the capture the hysteresis experiment used, and
a whole race rather than part of one.

The recording was still running when this was written, so re-running the
extractors will pick up more frames than the note's numbers reflect. The ground
truth in `ground_truth.py` covers frames 0–4774; anything past that is unlabelled
and the evaluation truncates to the shorter of the two.

## Order of operations

```bash
B=/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Iowa_Corn_350

# 1. Signals. Visual ~8 min, furniture ~8 min, audio ~2 min.
uv run python experiments/structure/extract_visual.py    --dir "$B" --out experiments/structure/visual.jsonl
uv run python experiments/structure/extract_furniture.py --dir "$B" --out experiments/structure/furniture.jsonl
uv run python experiments/structure/extract_audio.py     --dir "$B" --out experiments/structure/audio.jsonl

# 2. Ground truth (already verified; this just materialises truth.json).
uv run python experiments/structure/ground_truth.py

# 3. Analysis.
uv run python experiments/structure/analyse_structure.py   # durations, spot grid, black frames
uv run python experiments/structure/audio_probe.py         # how much audio alone carries
uv run python experiments/structure/evidence.py            # fused out-of-fold p(ad) per frame
uv run python experiments/structure/evaluate.py            # evidence x policy table
uv run python experiments/structure/sensitivity.py         # is the state machine on a plateau

# 4. Cross-broadcast, against server/frames/.
uv run python experiments/structure/cross_check_frames.py  # furniture vs the Fox manual labels
uv run python experiments/structure/check_corrections.py   # furniture vs the operator's error set
uv run python experiments/structure/cross_check_audio.py   # audio trained on NBC, tested on FS1
```

`evidence.py` must run before `evaluate.py` and `sensitivity.py`.

## Speech

Uses the local faster-whisper checkout at `~/Code/projects/faster-whisper-py`,
not this project's venv, so it is invoked with that interpreter directly.

```bash
S=/tmp   # scratch for the chunk wav
~/Code/projects/faster-whisper-py/venv/bin/python \
    experiments/structure/transcribe_all.py --tmp $S --threads 20
uv run python experiments/structure/transcript_cues.py   # which phrases precede a break
uv run python experiments/structure/cue_policy.py        # what a cue is worth as a trigger
```

`transcript_cues.py` and `cue_policy.py` both truncate to whatever the
transcript covers, so they are useful while `transcribe_all.py` is still
running — just re-run them when it finishes.

`audio_track.py` rebuilds a continuous PCM track from the overlapping per-frame
clips — take the last `t[i]-t[i-1]` seconds of each and they tile the timeline
exactly once. That is worth knowing independently of speech: it means any
whole-broadcast audio analysis is possible from the saved clips.

**This runs on CPU at about 2x realtime, so the full broadcast takes ~75
minutes.** The box has an NVIDIA card and a loaded driver, but `/dev/nvidia*` is
not reachable from the Claude Code sandbox, so ctranslate2 reports no CUDA
device. If that gets fixed, this becomes a couple of minutes.

## Ground truth, and how to trust it

`ground_truth.py` carries the verified segment list and documents how it was
built. The short version: an eyeball pass and a signal-derived pass disagree in
opposite directions, and every frame where they disagreed was reviewed
individually.

The helper scripts used to produce and check it are kept:

- `sheet.py` — contact sheets over a frame range
- `verify_boundaries.py` — one row per boundary, the frames either side
- `sheet_disputes.py` — one row per eyeball/signal disagreement
- `refit_truth.py` — the furniture-derived boundary proposal

## Auditing the ground truth

`ground_truth.py` documents how the labels were built, but the labels were
assigned by reading contact sheets rather than by the operator, so the numbers
here are only as good as that reading.
`experiments/review_ground_truth.py` puts each labelled frame back on screen
next to every independent signal bearing on it, and records a human ruling per
frame in `review_verdicts.json` — kept separate from the experiment data so the
audit and the thing audited never mix.

A ruling says what the frame *is* (`ad`, `content`, `other`), not whether the
label was right, because agreement follows from that and not the reverse:
"disagree" on an `ad` label never says whether the reviewer meant content or one
of the bumper and sponsor-billboard cases the labelling rule declines to decide.
Rulings carry the label they were made against, and because the captures overlap
the app shows any ruling made on the same frame while reviewing the other
dataset, so ruling one frame two ways shows up instead of hiding in the JSON.

```bash
uv run python experiments/review_ground_truth.py     # http://localhost:8766/
```

Two filters find the frames actually worth a human's time:

- **Anchor conflicts** (27) — the USA/peacock bug was detected but the frame is
  labelled `ad`. All 27 sit within nine frames of a break edge and score 0.91+
  against a 0.65 threshold, so the bug is unmistakably on screen. Either the
  edges are placed a few frames wide, or the bug genuinely survives into the
  opening wipe — which the `nonstop` convention in `ground_truth.py` already
  counts as part of the break. The distinction needs eyes, not more code.
- **Cross-experiment conflicts** (64) — see below.

The hysteresis experiment's continuous capture is a *prefix of this same
recording*, so 1572 frames carry two labels produced by two independent passes.
They agree on 95.93% and disagree on 64 frames, every one of them in the same
direction: hysteresis says `content`, this experiment says `ad`. The
disagreements are not boundary jitter but four contiguous runs, two of them
about a minute long (frames 111–142 and 275–301). Those two runs carry no
OpenCV anchor either way, so nothing but a human decides them.

That 4% is the honest uncertainty band on any label-derived number here, and it
is wider than the gaps between the top policies in `results.json` — several of
which differ by less than a tenth of a point.

## Gotchas

- **Never split these clips at random.** They are 4 s long and arrive every 2 s,
  so consecutive rows share half their samples. Every split in `audio_probe.py`
  and `evidence.py` is by segment; `audio_probe.py` prints the leaky random-split
  numbers alongside for comparison so the size of the effect stays visible.
- `ProcessPoolExecutor` cannot start under the sandbox (it fails creating a
  socketpair), so `extract_audio.py` uses threads. The FFT work releases the GIL,
  so this costs little.
- `furniture.jsonl` may be longer than `visual.jsonl` if the recording grew
  between runs. Everything joins on filename, so that is harmless.
