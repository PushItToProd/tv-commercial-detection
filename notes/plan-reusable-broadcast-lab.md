# A reusable broadcast lab

## Context

The current experiment tree measured a lot but can only ever measure one race.
`experiments/structure/` has exactly one slot for one broadcast's artifacts — a
second capture would overwrite the first — and its ground truth is a Python
literal of *frame indices* (`ground_truth.COARSE`) that is meaningless for any
other recording and shifts if a single image is missing.
`review_ground_truth.py` hardcodes one absolute broadcast path and three dataset
names, and holds ~780 lines of untested JavaScript inside a Python string.

Meanwhile the data has moved on:

- `Autotrader 400` on Fox — **8,106 frames, 3.8 GB, already on disk and
  unlabeled.** This is recommendation 5 of `notes/broadcast-structure-2026-08.md`
  ("record a continuous broadcast on Fox") already satisfied, and it unlocks the
  Fox-only `rectangle_match` path that nothing has ever evaluated.
- `Iowa Corn 350` has grown from the 4,775 frames reviewed to **5,974**.
  Incremental processing is a live requirement, not a hypothetical.
- Two ~929-frame RACER Network captures, plus two 1-frame directories created by
  title flicker.

The goal is a tool that treats "a broadcast" as data rather than as a constant:
point it at `record_broadcast.py` output, process it incrementally, label it in
the same UI, and eventually score sensors and policies across all of them with a
held-out split. The held-out split is the point — every number in the notes so
far was tuned on the one broadcast it was scored on.

The binding constraint is not code. It is **one human's labeling time**, and
everything here is judged by whether it shortens that.

**Milestone 1 (this plan): identity, cuts, processing, labeling app.** The
evaluation harness is designed for here but built second, once there is a second
labeled broadcast to hold out.

---

## The data model

Three ideas carry the whole design, and each replaces something in the
experiment tree that was true only of Iowa.

### Identity: video ID, pass, offset

`page_url` already carries a stable program ID, and it is a better key than
anything currently used:

| directory | video ID | frames |
|---|---|---|
| `Oregon-s_FOX_Autotrader_400` | `XyIC6SWbJK0` | 8,106 |
| `USA_4K_Iowa_Corn_350` | `1LaATJR0CeM` | 5,974 |
| `RACER_Network_Launch_Control` | `NMiMk5Ge4LQ` | 928 |
| `USA_4K_NASCAR_Cup_Series_-_YouTube_TV` | `NMiMk5Ge4LQ` | **1** |
| `RACER_Network_2026_Icelandic_Off-Road` | `ZQzTSHtZWAI` | 929 |
| `RACER_Network_NASCAR_Cup_Series_-_YouTube_TV` | `ZQzTSHtZWAI` | **1** |

Both 1-frame "title flicker" stubs are the *same program* as a real capture,
recorded one frame before `video_title` populated. Keying on video ID merges
them automatically. The frame-count floor and the deferred `record_broadcast.py`
change both disappear.

So: **program** is the video ID, **pass** is one capture session, **position** is
`video_offset`. A program may have several passes — the reboot-mid-race case, or
a deliberate re-capture at a finer cadence — and they share one timeline.

`video_offset` is a clean playback clock. Measured over the existing captures it
drifts from wall clock by 0.09 s across Oregon's 4.5 hours (max excursion
0.44 s) and 0.014 s across Iowa's 3.3 hours. Iowa's first frame sits at offset
160.99, i.e. capture started 2:41 into the program, so offsets are already
program-relative rather than capture-relative.

Two properties of offsets the code must respect, both already present in the
data:

- **Not monotonic.** The Icelandic capture contains a backward seek: one
  inter-frame delta of −10.04 s followed by +12.03 s. Two frames can share an
  offset.
- **Not on a grid.** Inter-frame deltas run 1.60–2.11 s against a nominal 2.0. A
  1 s pass will not share a single exact offset with a 2 s pass.

Therefore offset is a **coordinate, not a key**. The frame filename stays an
opaque, stable handle; offset is the axis that places frames from any pass on one
timeline.

**A verification gap:** the reboot-and-resume case requires `currentTime` to be
program-relative rather than live-edge-relative, and the existing captures cannot
distinguish the two (both hypotheses fit continuous playback). The extension is
being changed to record `video.duration` and `video.seekable.start(0)`/`end(0)`;
for a live stream `duration` is `Infinity`. Until a capture carries those fields,
treat multi-pass as designed-for but unproven, and do not plan a capture around
resuming one.

### Ground truth: cuts on the offset axis

A segmentation is fully specified by an initial label and an ordered list of
cuts. Iowa is **34 cuts**, not 4,775 labels.

This is not a compression of the old model, it is a different object, and it is
the one the operator's workflow already produces. Collapsing the 4,775 stored
verdicts into runs, with `other` absorbed into its surrounding run, yields
**exactly 35 runs / 34 cuts — matching `truth.json`'s 35 segments exactly.**

Cuts also make multi-pass work with no matching logic at all. Labeling any pass
at any cadence is evaluating a piecewise-constant function at new sample points.
There is no nearest-neighbour join, no tolerance parameter, and a finer pass can
*refine* cut placement rather than needing to be relabeled — every cut today is
only known to ±2 s because that is the sample rate.

A cut therefore records its bracket, not just a point:

```json
{"at": 71.4, "lo": 70.4, "hi": 72.4, "to": "ad", "by": "operator", "note": ""}
```

`lo`/`hi` are the offsets of the two frames it lies between; `at` defaults to
their midpoint. A finer pass narrows the bracket.

### Verdicts are binary; cost lives in annotations

`other` is retired as a verdict. It was carrying two orthogonal things at once:
what the frame *is*, and how much its being misclassified matters. The stored
data already shows the conflation — of the 124 `other` frames, `care_back` is 0
on 123 and `care_away` is 0 on 78, and the operator's own notes say *"technically
content but not necessarily super important"*. Meanwhile `risk: fraught` already
crosses the verdict boundary: of 95 fraught frames, 46 carry a confident `ad`
verdict.

The decisive argument is measurement. Post-break ad reads — racing-looking
footage that is really advertising — are the material most worth studying, and a
frame labeled `other` cannot be scored against anything. It drops out of every
metric precisely when you want to measure it. Give it a truth value and the miss
becomes measurable; put "I expect classifiers to fail here" into `risk` and
`care_*` and you get three reportable numbers instead of one muddled one:
accuracy on `safe`, accuracy on `fraught`, and care-weighted cost.

The data confirms this costs nothing: **no `other` frame ever forms a segment of
its own** — absorbing them into their surrounding run reproduces the 35-segment
truth exactly.

One narrow escape hatch survives, as a scoring mask rather than a class: an
`exclude` list for frames where the truth genuinely is undecidable (the 2 bumper
and 2 silence frames, corrupt captures, the 4 `artifact` frames).
`reeval_manual.py` already has the `other="exclude"` mode, so the machinery
exists.

### Annotations are intervals

The operator's bulk-note workflow is already interval annotation written in
prose (*"this whole run of clips up to X is a pre-race hype segment"*). Facets
follow the same shape: collapsing the 4,775 facet records into runs of constant
value gives **261 intervals** — an 18× reduction, and 129 if restricted to
frames carrying any non-default value.

So annotations are `[{from, to, facets, note}]` on the offset axis, with
single-frame notes as a degenerate interval. `where`
(`interior`/`pre_break`/`onset`/`rejoin`) stops being stored at all: it is a
pure function of distance to the nearest cut, and no longer tied to
`EDGE = 20 s / DT` frame counts.

---

## Decisions taken

- **Labels are git-tracked**, signals are not. Cuts and annotations are small,
  hand-made and irreplaceable → `server/labels/<video_id>/`. Signal extracts are
  tens of MB and regenerable → `<pass>/lab/`, next to the media. (`/mnt/data` is
  now in the sandbox write allowlist.)
- **`experiments/` is not touched.** It backs two published notes. The lab is
  built fresh. `policies.py`, `evaluate.score` and the extractor *math* are
  copied with their comments intact — roughly 250 lines that took measurement to
  get right and have no better version.
- **`review_ground_truth.py` is rebuilt, not ported.** Its 1,841 lines are shaped
  around a dataset-plus-overlap model that this plan retires, and four of its
  eighteen filters exist only because two experiments labeled the same frames
  twice. Its *interaction contract* is preserved and is written down below; none
  of its code is.
- **`facets.py`'s inference layer is spent.** `NOTE_PATTERNS`, `CARE_LOW/HIGH`
  and `infer_from_note` mined prose the operator had already written, against
  "the operator's own wording, including the variants actually written." There is
  no prose to mine on Fox. Keep its 264 `inferred` outputs, keep the schema and
  the derivation-from-cuts; leave the regexes frozen in `experiments/`.
- **Processing proposes cuts.** Labeling becomes accept-and-nudge rather than
  label-from-scratch. Proposals never write into `cuts.json`.

---

## Layout

```
server/src/tv_commercial_detector/lab/
  program.py        Program + ProgramLibrary: video-ID identity, passes, one timeline
  cuts.py           CutList: initial label + ordered cuts; segment/label derivation
  annotations.py    Interval annotations and the exclude mask
  labels.py         LabelStore: cuts + annotations, mtime-cached, atomic, locked
  proposals.py      Propose cuts from anchors
  process.py        Incremental extraction driver
  cli.py            lab ls | process | propose | label | doctor | import-legacy
  signals/
    base.py         Signal protocol: name, version, params, warmup, needs, extract()
    visual.py       Profile-aware template scores + frame statistics
    furniture.py    Template-free edge persistence
    audio.py        Per-clip DSP features
  label_app/
    app.py          FastAPI factory (thin)
    api.py          Routes
    templates/index.html
    static/*.css, *.mjs

server/labels/<video_id>/
  meta.json         title(s), network, profile, green_flag (offset), split, passes
  cuts.json         {"initial": "content", "cuts": [{at, lo, hi, to, by, note}]}
  annotations.json  [{from, to, facets: {...}, note}]
  exclude.json      [{from, to, why}]

<pass>/lab/                       (on /mnt/data, gitignored, regenerable)
  signals/<name>.jsonl            one record per frame, keyed by filename
  signals/<name>.meta.json        {version, params, count, updated}
  proposals.json
```

`lab` deps go in a new `[dependency-groups] lab` in `server/pyproject.toml`
(`scipy` for the audio DSP — already present transitively via `imagehash`, but
name it explicitly; `numpy` arrives with opencv). `uv sync --group lab`. The CLI
needs a `[project.scripts] lab = "..."` entry, which `pyproject.toml` does not
have today.

---

## 1. Program library — `program.py`

**`Pass`** wraps one `record_broadcast.py` directory: `root`, `images_dir`,
`audio_dir`, and `frames()` reading `classifications.jsonl` sorted by timestamp.
Sorting is load-bearing, not defensive — the Icelandic capture is genuinely out
of order on disk.

**`Program`** groups passes sharing a video ID and presents one timeline:

- `frames()` — every frame from every pass, ordered by offset, each carrying its
  `pass_id`, `filename` and `offset`.
- `dt` — **measured** median inter-frame gap, not the `DT = 2.0` constant
  hardcoded at `experiments/structure/evaluate.py:14`. Measured: 2.003 s across
  all four captures.
- `meta` — from `server/labels/<video_id>/meta.json`: `profile`, `green_flag`,
  `split`, pass list.

`green_flag` is stored **as an offset**, not a frame index. The current value
lives at `experiments/facets.py:127` as `GREEN_FLAG = 475`, an index; on Iowa
that is offset **1113.42** (952.43 rebased to the first frame). Keeping it an
index reproduces exactly the fragility this plan condemns in `COARSE` — it
survives Iowa's growth today only because the 1,199 new frames appended as a
clean suffix (verified: the labeled 4,775 are precisely indices 0–4774).

**`ProgramLibrary(root)`** scans `<root>/<host>/<dir>/`, reads one line of each
`classifications.jsonl` for the video ID, and groups. Directories with no
readable video ID are reported by `lab ls` rather than silently skipped.

## 2. Cuts and annotations — `cuts.py`, `annotations.py`, `labels.py`

**`CutList`** is the ground truth: an initial label plus ordered cuts. It
derives, on demand and never stored:

- `segments()` — gapless by construction, so the whole class of overlap and hole
  bugs cannot occur.
- `label_at(offset)` and `labels_for(frames)` — the piecewise-constant
  evaluation that makes multi-pass free.
- `distance_to_cut(offset)` — what `where` is computed from.

Invariants enforced on write: cuts strictly increasing in `at`, alternating
labels (two consecutive `→ ad` cuts is a bug, not a segmentation), and every
`lo < at < hi`.

**`LabelStore`** replaces the load-mutate-save-whole-file pattern at
`review_ground_truth.py:278-286`, which has no locking and silently loses one of
two concurrent updates. Keep the atomic `.tmp` + `Path.replace()`; add an mtime
cache and a per-file lock around read-modify-write. (Decide explicitly whether
that lock is the `filelock` package — a new dependency — or hand-rolled; v1 of
this plan said "filelock-style" and named neither.) Port `apply_note` /
`NOTE_MODES` / `merge_preserving` semantics verbatim.

## 3. Incremental processing — `process.py`, `signals/`

A `Signal` declares `name`, `version`, `params`, `needs` (`image` / `audio`) and
`warmup` — how many preceding frames it must see to produce a correct value.

Per pass × signal:

1. Read `<name>.meta.json`. If `version` or `params` differ from the code's
   current values, discard and recompute in full.
2. Read existing `<name>.jsonl` keys; find the first frame not yet present.
3. **Prime the window**: re-read the `warmup` frames *before* that point, compute
   through them, discard their output, and append from the first genuinely new
   frame.
4. Append, then rewrite the meta sidecar.

Step 3 is the part that is easy to get wrong and the reason this is not a plain
"skip what exists" loop. `furniture` intersects Canny edges over 4- and 12-frame
windows (`warmup=12`); `visual` carries `mad_prev` / `hist_corr_prev` /
`phash_dist_prev` (`warmup=1`); `audio` is per-clip (`warmup=0`). Without
priming, a resumed run writes a wrong value at every resume point — and resume
points land mid-broadcast, in exactly the boundary regions policies are judged
on.

Signals are computed **per pass**, keyed by filename, because they are properties
of the captured media. Only labels live on the program timeline.

Safe against a live recording: `record_broadcast.py` only appends to
`classifications.jsonl` under a lock, so a mid-broadcast run sees a shorter file.

**Signals:**

- `visual` — dispatches on `meta.profile`. For `nascar_on_nbc`, the real
  `peacock_score` / `usa_score` / `side_by_side_score`. For `nascar_on_fox`, the
  network and side-by-side logo checks **plus
  `classification.rectangle_match.image_has_known_ad_rectangle`** — Fox-only, and
  never scored by any experiment. Plus profile-independent statistics (phash,
  deltas, letterbox, `black_frac`, `ticker_edges`). **Needs a defined
  profile-independent-only path**: neither RACER capture is NASCAR on a known
  network, and v1 of this plan left their `meta.profile` undefined.
- `furniture` — unchanged in substance, `W_SHORT` / `W_LONG` as declared params.
- `audio` — unchanged; keeps `ThreadPoolExecutor` (`ProcessPoolExecutor` cannot
  create a socketpair under the sandbox).

Anchor thresholds (`PEACOCK_TH` 0.55, `USA_TH` 0.65, `SBS_TH` 0.8, hardcoded in
`experiments/structure/timeline.py`) move into a per-profile table: they are
properties of a profile's templates, not of a broadcast.

Milestone 1 builds `visual` only. `furniture` and `audio` are evaluation inputs
and do not shorten labeling time; they come with the eval harness.

```bash
uv run lab process --all
uv run lab process 1LaATJR0CeM
uv run lab process XyIC6SWbJK0 --signal visual --force
```

## 4. Labeling app — `label_app/`

**The primary view is the timeline, not the card.** Iowa is 34 boundary
decisions against 4,775 frame decisions; Fox will be roughly 55. The existing
tool is card-first with the contact sheet secondary, which is backwards for the
one constraint that matters.

The core loop, which is the workflow the operator already runs by hand with
shift-click and bulk notes:

1. Scrub the contact sheet forward from the last cut.
2. Land on the **last frame of the current segment** and press `a` / `c`.
3. Everything from the previous cut through that frame takes the label; a cut is
   written with `lo` = that frame's offset, `hi` = the next frame's offset,
   `at` = midpoint.
4. The view jumps to the next unlabeled frame.

Everything else is secondary: a **refine** mode that steps frame-by-frame either
side of an existing cut to narrow its bracket, a **card** view for one frame's
signals, and an **annotate** mode for painting facets and notes over a range.
Per-frame notes remain, for annotation rather than for labeling.

Proposed cuts (§5) render as ghost markers the operator confirms with a
keystroke or drags; accepting one is the same action as placing one.

**The interaction contract to preserve** — this is what is worth keeping from the
old app, and it is exactly the list v1 nominated for JS tests: keyboard dispatch
precedence, range selection across page boundaries, note-mode application
(`replace` / `fill` / `append`), audio playback exclusivity, and the
`select`/`focusCard` shadowing class of bug (an inline handler puts the element
on the scope chain, so a global named `select` is shadowed by
`HTMLInputElement.prototype.select`). Write this down as a spec before deleting
anything; the rebuild is then a reimplementation against a known contract.

The frontend is files from the start — ES modules with delegated events, not
inline `onclick`. The "large behavioral risk for no benefit" argument against
delegation only applies when preserving an existing file's behavior, which a
rebuild is not doing.

**Filters declare what they need** and are hidden for programs that lack the
signal. The cross-dataset filters (`cross_conflict`, `contradicts`,
`model_wrong`, `model_flips`) are **retired, not ported** — they depended on the
hysteresis `gt` labels and `replay.json`, and on the `burst` dataset that points
at `server/frames/images`, the live save dir, which is not a capture at all. This
does lose the independent-second-opinion mechanism, and proposals do not replace
it: a proposal derived from your own anchors is not an independent pass.

New filters: `unlabeled` (after the last cut), `near-cut` (within the refine
window of any cut), and `fraught`.

**Migration.** `lab import-legacy`:

1. Reads `experiments/review_verdicts.json` (only the `structure` dataset carries
   verdicts — 4,775, same key set as facets) and collapses to cuts.
2. Emits the 124 `other` frames to a re-adjudication queue rather than guessing.
   Most are already implied — `spot_audio` (33) and `ad_read` (46) → `ad`,
   "technically content" → `content` — leaving perhaps 30–40 needing a real look.
3. Collapses `review_facets.json` into the 261 constant intervals, preserving
   `src`, `review` (41 frames) and `confirmed` (180). Note the file is
   `{schema, facets}`, so unwrap it; `schema` is code-derived and not migrated.
4. Is **re-runnable and merge-aware**, not one-shot. Review is live in the old
   tool right now (`review_facets.json` is modified in both index and working
   tree), so a one-shot import creates two writable copies of the same truth.
   `merge_preserving` already does the hard part. The old tool goes read-only at
   cutover.

## 5. Proposals — `proposals.py`

Propose **cuts**, not per-frame labels. Anchors give candidate boundaries
directly: a side-by-side banner appearing is a `→ ad` cut, a network bug
resuming is a `→ content` cut. Break durations cluster tightly enough
(NON STOP breaks 120–122 s, sd 0.7; full breaks 118–217 s; shortest interior
content run 94 s) that a proposed edge is usually within a frame or two of right,
which is what makes accept-and-nudge viable.

Writes `<pass>/lab/proposals.json`. Regenerable, so not in git, and **never
merged into `cuts.json`** — a cut is only ever created by a human action.

Deferred to the eval milestone: proposing cuts from a transferred audio model.
That was v1's cross-network transfer test, and it cannot be validated until Fox
has labels, which is what this milestone produces.

## 6. Evaluation — designed here, built next

Not in this milestone, but the layer above must not foreclose it:

- **Boundary timing is the headline metric,** not frame accuracy. With cuts as
  truth, score a policy on median and max seconds late at break onset and at
  rejoin, plus switches and flaps. `ad_shown` / `race_missed` / `lat_med` in
  `evaluate.py` are already this metric under other names. Frame accuracy is the
  number that makes a flapping policy look good: on Iowa, `stateless` scores
  0.985 accuracy while taking 68 switches with 23 flaps against a true 34.
- **Cost from care values.** `cost = Σ care_away` where showing race over an ad,
  `care_back` where showing the alternate input over racing, plus a flap penalty.
  Falls back to flat weights where annotations are absent. The current weighting
  (ad 3:1 against race, no flap term) is why it ranks a thrashing policy first.
- **Sensors × policies × programs.** `policies.py` is already pure
  (`fn(ev, key) -> list[str]`) and copies directly. Note that its dwell constants
  are **frame counts at a ~2 s cadence** (`NONSTOP_LEN=60`, `MIN_AD_DWELL=50`,
  `MIN_CONTENT_DWELL=40`), so with `dt` measured per program they must be
  expressed in seconds and converted, not merely parameterized. Four of the five
  are already keyword arguments on `hsmm`; only `CLAMP` is not.
- **Splits.** `meta.split` is `train` / `dev` / `test`; `lab eval` defaults to
  train+dev and refuses `test` without `--allow-test`. With ≥2 labeled programs,
  leave-one-program-out replaces within-broadcast segment-blocked CV as the
  headline number.
- **Write protection.** The eval path imports `LabelStore` read-only. An agent
  told to iterate toward a target must not be able to move the target.

## 7. Also worth doing

- **`lab doctor`** — per program: cadence gaps, offset backsteps (the Icelandic
  capture has one), silent audio clips (reuse `audio_health`),
  image/audio/jsonl count mismatches, duplicate phashes, proportion labeled,
  and whether `duration`/`seekable` fields are present. Iowa was collected for a
  whole summer with dead audio before anyone noticed; a one-command check is
  cheap.
- **Facet generation.** v1 listed a `facets.py` module but no verb that builds
  facets, which would have left Fox with 8,106 frames and nothing to weight them
  by. Ordering is: label → derive annotations from cuts → hand-correct. Needs
  `lab annotate --derive <video_id>`.
- **Committed tests.** Python under `server/tests/lab/`: library discovery and
  video-ID grouping, incremental processing **with the warmup path asserted
  against a full recompute**, version-bump invalidation, `CutList` invariants and
  `label_at` against the migrated Iowa truth, `LabelStore` concurrent-write
  safety, `merge_preserving` including deliberately-cleared values, and API
  routes via `TestClient`. A deno harness under `server/tests/lab/js/` (deno is
  already installed) covering the interaction contract listed in §4.

---

## Verification

```bash
cd server && uv sync --group lab

# 1. Identity: 4 programs, both title-flicker stubs folded into their real
#    capture by video ID, dt ~2.00s each
uv run lab ls

# 2. Migration, then confirm the cut model reproduces the old truth exactly
uv run lab import-legacy
uv run python -c "
import json
from tv_commercial_detector.lab.cuts import CutList
cuts = CutList.load('labels/1LaATJR0CeM/cuts.json')
segs = cuts.segments()
old = json.load(open('experiments/structure/truth.json'))['segments']
assert len(segs) == len(old) == 35, (len(segs), len(old))
print(f'{len(cuts.cuts)} cuts -> {len(segs)} segments, matching truth.json')"
# and every frame's derived label matches its stored verdict, except the 124
# `other` frames queued for re-adjudication
uv run lab import-legacy --report-other

# 3. Idempotence: full run, no-op re-run, then the resume path
uv run lab process 1LaATJR0CeM --signal visual
uv run lab process 1LaATJR0CeM --signal visual        # '0 new frames'
# truncate to 3000 rows, resume, and diff against the full run:
# the warmup path is correct iff every row matches byte for byte
uv run pytest tests/lab/test_process.py -k warmup

# 4. Fox: profile dispatch and the never-evaluated rectangle_match path
uv run lab process XyIC6SWbJK0 --signal visual
uv run lab doctor XyIC6SWbJK0

# 5. Proposed cuts on unlabeled data
uv run lab propose XyIC6SWbJK0
# expect: proposed cut count in the 40-60 range, anchor coverage %, and the
# median gap between consecutive proposed cuts against the known 118-217s /
# 94s+ break and content distributions

# 6. Label
uv run lab label            # http://localhost:8766/
```

In the browser, confirm on Iowa that all 34 cuts, every annotation and every
note survived migration and that the segment boundaries match `truth.json`; then
switch to Fox and label one hour of it, timing the pass. The success criterion
for this milestone is **the whole Fox broadcast labeled in a single sitting** —
if it is not, the tool has not solved the problem it was built for.

```bash
uv run pytest tests/lab/ -m "not integration"
deno test server/tests/lab/js/
uv run ruff check src/ && uv run ruff format --check src/
```

## Out of scope

Changes to `server/src/tv_commercial_detector/` proper. Adopting the audio sensor
and the `hsmm` policy in the live detector is a separate plan
(`notes/plan-adopt-audio-hsmm.md`) whose promotion gate is the held-out
broadcast this milestone produces.
