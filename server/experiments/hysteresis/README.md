# Temporal / hysteresis experiment

Scratch code behind `notes/temporal-hysteresis-2026-08.md`. Not wired into the
server and not needed to run it — kept so the numbers in that note can be
re-derived or re-run against a new broadcast.

Everything writes into this directory. The scripts take a few minutes each and
talk to llama.cpp at `gmktec.zane.network:3002`.

## Datasets

Two captures, for two different reasons.

**Burst archive** — `server/frames/` for 2026-08-09, the Iowa Corn 350 on USA.
1400 frames in the race window, but `recent_frames` is a 5-deep deque flushed
once a minute, so the archive is 208 disjoint ~8-second bursts covering 23% of
the broadcast. Bursts get written when the classifier changes its mind, so
transitions are over-represented roughly 4x. Useful for hard cases, useless for
anything needing more than 8 seconds of history.

**Continuous recording** — `record_broadcast.py` output, 1572 frames over 51
minutes at a 2.0 s cadence with no gaps. This is the one that can actually
exercise a temporal policy, and the one the conclusions rest on.

```bash
# Burst archive
uv run python extract_features.py --day 2026-08-09 --out frames_0809.jsonl
python build_dataset.py                      # + burst_labels.py, frame_labels.py
uv run python replay_pipeline.py --reps 3 --out replay.json
python evaluate.py

# Continuous recording
uv run python extract_cont.py --dir <recording-dir> --out cont.jsonl
python build_cont_dataset.py
uv run python replay_pipeline.py --reps 3 --dataset cont_dataset.jsonl \
    --images <recording-dir>/images --out cont_replay.json
python evaluate.py continuous
```

## Ground truth

Labelled by eye from contact sheets (`contact_sheet.py`, `sheet_cont.py`), not
by the classifier. On the continuous recording the two OpenCV checks anchor the
labels — verified 48/48 network-bug frames are live coverage and 16/16
side-by-side frames are breaks — and the 378 frames they leave undecided were
each reviewed. On the burst archive the labels were checked against the
operator's own `/report_wrong` corrections: 10/10 agreement.

`burst_labels.py` and `frame_labels.py` carry the labelling rule and, more
usefully, the cases where it does not decide.

## Files

| File | Purpose |
|---|---|
| `extract_features.py` / `extract_cont.py` | Per-frame OpenCV scores, phash, frame-to-frame diffs |
| `build_dataset.py` / `build_cont_dataset.py` | Join signals to ground truth, cut into episodes |
| `replay_pipeline.py` | Re-run the real `nascar_on_nbc` pipeline, N reps, with timings |
| `arms.py` | The temporal policies and the cost model |
| `evaluate.py` | Scores every policy; `continuous` argument picks the dataset |
| `llm_context_arms.py` | Live arms that feed the model previous frames and/or textual history |
| `contact_sheet.py` / `sheet_cont.py` | Contact sheets for hand labelling |
| `probe_llm.py` / `probe_llm2.py` | llama.cpp latency and token cost by image count |
| `probe_usa_corner.py` | Whether a lower-right USA logo is a usable ad signal |

`probe_llm.py` is kept only as a warning: it reused one image across reps, so
llama.cpp's prompt cache served every call after the first and the timings were
meaningless. `probe_llm2.py` uses a fresh frame per rep, which is what
production sees.
