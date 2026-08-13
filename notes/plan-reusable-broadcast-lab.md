# A reusable broadcast lab

## Context

The current experiment tree measured a lot but can only ever measure one race.
`experiments/structure/` has exactly one slot for one broadcast's artifacts — a
second capture would overwrite the first — and its ground truth is a Python
literal of *frame indices* (`ground_truth.COARSE`, imported by six files) that is
meaningless for any other recording and shifts if a single image is missing.
`review_ground_truth.py` hardcodes one absolute broadcast path and three dataset
names, and holds ~780 lines of untested JavaScript inside a Python string.

Meanwhile the data has moved on:

- `Oregon-s_FOX_Autotrader_400` — **8,106 frames, 3.8 GB, already on disk and
  unlabeled.** This is recommendation 5 of `notes/broadcast-structure-2026-08.md`
  ("record a continuous broadcast on Fox") already satisfied, and it unlocks the
  Fox-only `rectangle_match` path that nothing has ever evaluated.
- `USA_4K_Iowa_Corn_350` has grown from the 4,775 frames reviewed to **5,974**.
  Incremental processing is a live requirement, not a hypothetical.
- Two ~929-frame RACER Network captures, plus two 1-frame directories created by
  title flicker when `video_title` is briefly empty.

The goal is a tool that treats "a broadcast" as data rather than as a constant:
point it at a `record_broadcast.py` directory, process it incrementally, label it
in the same UI, and eventually score sensors and policies across all of them with
a held-out split. The held-out split is the point — every number in the notes so
far was tuned on the one broadcast it was scored on.

**Milestone 1 (this plan): data layer, processing, review app.** The evaluation
harness is designed for here but built second, once there is a second labeled
broadcast to hold out.

## Decisions taken

- **Labels are git-tracked**, signals are not. Verdicts and facets are small,
  hand-made and irreplaceable → `server/labels/<slug>/`. Signal extracts are tens
  of MB and regenerable → `<broadcast>/lab/`, next to the media.
- **`experiments/` is not touched.** It backs two published notes. The lab is
  built fresh; `policies.py`, `evaluate.score` and the extractors are ported by
  copy-and-generalize, not by import. Accepting the duplication is the price of
  keeping the published numbers reproducible from frozen code.
- **Processing proposes labels.** Review becomes confirm-and-correct rather than
  label-from-scratch. Proposals never write into `verdicts.json`.
- **Per-frame verdicts *are* the ground truth.** Segments are derived from them
  (the collapse already exists as `experiments/structure/reeval_manual.py:81`),
  which retires `ground_truth.COARSE` and its frame-index fragility for good.

---

## Layout

```
server/src/tv_commercial_detector/lab/
  broadcast.py        Broadcast + BroadcastLibrary: discover, index frames, resolve paths
  labels.py           LabelStore: verdicts + facets, mtime-cached, atomic, locked
  facets.py           Facet schema; per-broadcast green_flag instead of a constant
  proposals.py        Prelabel a broadcast from anchors + a transferred audio model
  process.py          Incremental extraction driver
  cli.py              lab ls | process | propose | review | doctor | import-legacy
  signals/
    base.py           Signal protocol: name, version, params, warmup, needs, extract()
    visual.py         Profile-aware template scores + frame deltas
    furniture.py      Template-free edge persistence
    audio.py          Per-clip DSP features
  review/
    app.py            FastAPI factory (thin)
    api.py            Routes
    templates/index.html
    static/app.css, app.js

server/labels/<slug>/
  meta.json           profile, network, green_flag, split, part dirs, notes
  verdicts.json       filename -> {verdict, note, at}
  facets.json         filename -> {video, audio, phase, where, care_*, risk, src}

<broadcast>/lab/                      (on /mnt/data, gitignored, regenerable)
  signals/<name>.jsonl                one record per frame, keyed by filename
  signals/<name>.meta.json            {version, params, count, updated}
  proposals.json
```

`lab` deps (`scipy`; numpy already arrives with opencv) go in a new
`[dependency-groups] lab` in `server/pyproject.toml`, so production installs stay
unchanged. `uv sync --group lab`.

---

## 1. Data layer — `broadcast.py`, `labels.py`

**`Broadcast`** wraps one `record_broadcast.py` directory:

- `slug` (directory name), `root`, `images_dir`, `audio_dir`
- `frames()` — reads `classifications.jsonl`, sorts by `timestamp`, returns rows
  with an added `i` and `t` (`video_offset` rebased to the first frame). This is
  the single ordering authority; nothing else sorts.
- `dt` — **measured** median inter-frame gap, not the `DT = 2.0` constant
  hardcoded in `experiments/structure/evaluate.py:14`. Every seconds-valued metric
  derives from it.
- `meta` — from `server/labels/<slug>/meta.json`: `profile` (which classifier
  profile's signals apply), `green_flag` (the operator-supplied constant now at
  `experiments/facets.py:127`, one per broadcast), `split`, optional `parts`.

**`BroadcastLibrary(root)`** scans `<root>/<host>/<slug>/`, skipping directories
below a frame-count floor (default 5) so the two 1-frame title-flicker stubs stay
out of the way without being deleted. `meta.parts` lets a logical broadcast span
several directories (the pre-race show rolling into the race under a new title) —
frames concatenate in timestamp order and the slug of the first part names it.

**`LabelStore`** replaces the load-mutate-save-whole-file pattern at
`review_ground_truth.py:278-286`, which has no locking and silently loses one of
two concurrent updates. Keep the atomic `.tmp` + `Path.replace()`; add an mtime
cache (every read endpoint currently re-reads both JSON files from disk) and a
per-file `filelock`-style guard around read-modify-write. Verdicts lose their
dataset key — they are per-broadcast files now — and keep `judged`. Port
`apply_note` / `NOTE_MODES` / `merge_preserving` semantics verbatim; the
edit-preserving merge is what protects hand edits when facets are regenerated.

## 2. Incremental processing — `process.py`, `signals/`

A `Signal` declares `name`, `version`, `params`, `needs` (`image` / `audio`),
and `warmup` — how many preceding frames it must see to produce a correct value.

The driver, per broadcast × signal:

1. Read `<name>.meta.json`. If `version` or `params` differ from the code's
   current values, discard the output and recompute in full.
2. Read the existing `<name>.jsonl` keys. Take `frames()` and find the first
   frame not yet present.
3. **Prime the window**: re-read the `warmup` frames *before* that point, compute
   through them, discard their output (already written), and append from the
   first genuinely new frame onward.
4. Append, then rewrite the meta sidecar.

Step 3 is the part that is easy to get wrong and the reason this is not a plain
"skip what exists" loop. `furniture` intersects Canny edges over 4- and 12-frame
windows (`warmup=12`); `visual` carries `mad_prev` / `hist_corr_prev` /
`phash_dist_prev` (`warmup=1`); `audio` is per-clip (`warmup=0`). Without
priming, a resumed run silently writes a wrong value at every resume point — and
because runs resume mid-broadcast, those points land in exactly the boundary
regions the policies are judged on.

Safe against a live recording: `record_broadcast.py` only ever appends to
`classifications.jsonl` under a lock, so a mid-broadcast run just sees a shorter
file.

**Signals**, ported and generalized from the existing extractors:

- `visual` — dispatches on `meta.profile`. For `nascar_on_nbc`, the real
  `peacock_score` / `usa_score` / `side_by_side_score` (already the production
  code path via `experiments/structure/extract_visual.py:102`). For
  `nascar_on_fox`, the network and side-by-side logo checks **plus
  `classification.rectangle_match.image_has_known_ad_rectangle`** — Fox-only,
  and no experiment has ever scored it. Plus the profile-independent frame
  statistics (phash, deltas, letterbox, `black_frac`, `ticker_edges`).
- `furniture` — unchanged in substance, `W_SHORT` / `W_LONG` as declared params.
- `audio` — unchanged; keeps `ThreadPoolExecutor` (`ProcessPoolExecutor` cannot
  create a socketpair under the sandbox).

Anchor thresholds (`PEACOCK_TH`, `USA_TH`, `SBS_TH`, hardcoded at
`experiments/structure/timeline.py`) move into a per-profile table, since they
are properties of a profile's templates, not of a broadcast.

```bash
uv run lab process --all                 # every broadcast, every signal, resumable
uv run lab process USA_4K_Iowa_Corn_350  # one broadcast
uv run lab process --signal audio --force
```

## 3. Proposals — `proposals.py`

For a broadcast with no labels: anchors first (side-by-side banner → `ad`,
network bug → `content`), then an audio score from a model **trained on another
broadcast** (`--audio-model-from USA_4K_Iowa_Corn_350`), then the `hsmm` policy
to smooth per-frame guesses into runs — because rulings are made in runs and a
proposal that flickers is worse than none.

Writes `<broadcast>/lab/proposals.json` (`{filename: {label, confidence,
source}}`). Regenerable, so not in git, and **never merged into `verdicts.json`**
— a verdict is only ever created by a human action in the review UI.

This is also the first honest cross-network test of the audio sensor: the note's
transfer check had to lean on proxy labels and an easy subset, and a Fox
broadcast labeled with Iowa's audio model as the starting point measures it
directly.

## 4. Review app — `review/`

Same UI, same keybindings, same query-string contract. Three changes:

**Broadcasts replace datasets.** The dropdown is populated from
`BroadcastLibrary`, not the three hardcoded `<option>` labels with baked-in frame
counts at `review_ground_truth.py:933-935`. `/image/{slug}/{filename}` and
`/audio/{slug}/{filename}` resolve through the `Broadcast`, so the
`Path(images_dir).parent / "audio"` assumption goes away.

**Filters declare what they need.** Today `FILTERS` indexes structure-shaped keys
with `[]`, which forces `_load_hysteresis` to stuff dummy values (`conflict:
False`, `boundary_dist: 999`) into rows where they mean nothing. Instead each
filter declares required signal fields and is hidden for broadcasts that lack
them. New: `unconfirmed` (has a proposal, no verdict) and `uncertain` (low
proposal margin, or within the boundary window of a proposed run edge) — these
two are the throughput story. `counts_for` currently evaluates all 18 filters
over every row on every request; compute counts once per (broadcast, labels
mtime) instead.

**The frontend becomes files.** `_HTML` (`review_ground_truth.py:747-1818`) splits
into `templates/index.html`, `static/app.css` (~175 lines) and `static/app.js`
(~780 lines), along the `// ──` section banners already in the file. Keep them as
classic scripts with the existing inline `onclick` handlers — converting to ES
modules would mean rewriting every handler as delegation, which is a large
behavioral risk for no benefit here. The two templated values (`__SCHEMA__`,
`__DEFAULT_DATASET__`) become a `<script type="application/json">` config block.

The point of extracting the JS is that it becomes testable. It is untestable
today for a structural reason: it only exists inside a Python string literal, so
reaching it requires uvicorn and a browser.

**Migration.** `lab import-legacy` reads `experiments/review_verdicts.json` and
`review_facets.json` and writes `server/labels/USA_4K_Iowa_Corn_350/`. Only the
`structure` dataset carries verdicts, and `cont` is a prefix of that same
recording, so the 64 cross-experiment disagreements resolve by construction —
one frame, one file, one truth. Read-only on the originals; they stay put.

## 5. Evaluation harness — designed here, built next

Not in this milestone, but the layer above must not foreclose it:

- **Sensors × policies × broadcasts.** `policies.py` is already pure
  (`fn(ev, key) -> list[str]`) and ports directly; its five dwell/CUSUM constants
  become explicit params rather than module constants, since they were tuned on
  Iowa.
- **Cost from care values.** `cost = Σ care_away(f)` where showing race over an
  ad, `care_back(f)` where showing the alternate input over racing, plus a flap
  penalty. Falls back to flat weights where facets are unreviewed. This retires
  the three-way `other` problem: the cost needs no label at scoring time.
- **Splits.** `meta.split` is `train` / `dev` / `test`; `lab eval` defaults to
  train+dev and refuses `test` without `--allow-test`. With ≥2 broadcasts,
  leave-one-broadcast-out replaces within-broadcast segment-blocked CV as the
  headline number.
- **Write protection.** The eval path imports `LabelStore` read-only. An agent
  told to iterate toward a target must not be able to move the target.

## 6. Also worth doing

- **`lab doctor`** — per broadcast: cadence gaps, silent audio clips (reuse
  `audio_health`), image/audio/jsonl count mismatches, duplicate phashes,
  proportion labeled. The Iowa capture was collected for a whole summer with dead
  audio before anyone noticed; a one-command check is cheap.
- **Committed JS tests.** A deno harness under `server/tests/lab/js/` stubbing
  DOM + fetch, covering the pieces that broke during the build: keyboard dispatch
  precedence, range selection across page boundaries, note-mode application,
  audio exclusivity, and the `select`/`focusCard` shadowing class of bug (an
  inline handler puts the element on the scope chain, so a global named `select`
  is shadowed by `HTMLInputElement.prototype.select`).
- **Python tests** — `server/tests/lab/`: library discovery incl. stub-dir
  skipping, incremental processing **with the warmup path asserted against a
  full recompute**, version-bump invalidation, `LabelStore` concurrent-write
  safety, `merge_preserving` including deliberately-cleared values, filter
  predicates, and API routes via `TestClient`.
- **Stub directories.** Skip by frame-count floor rather than deleting. Worth a
  follow-up on `record_broadcast.py` so it does not create a directory until a
  title has held for a few frames — but that is a recorder change, out of scope
  here.

---

## Verification

```bash
cd server && uv sync --group lab

# 1. Discovery: 4 real broadcasts, 2 stubs skipped, dt ~2.01s each
uv run lab ls

# 2. Migration, then confirm nothing was lost
uv run lab import-legacy
uv run python -c "
import json
old=json.load(open('experiments/review_verdicts.json'))['structure']
new=json.load(open('labels/USA_4K_Iowa_Corn_350/verdicts.json'))
assert all(new[k]['verdict']==v['verdict'] for k,v in old.items()), 'verdict drift'
print(len(old),'verdicts migrated')"

# 3. Idempotence: full run, no-op re-run, then the resume path
uv run lab process USA_4K_Iowa_Corn_350 --signal furniture
uv run lab process USA_4K_Iowa_Corn_350 --signal furniture   # '0 new frames'
# truncate to 3000 rows, resume, and diff against the full run:
# the warmup path is correct iff every row matches byte for byte
uv run pytest tests/lab/test_process.py -k warmup

# 4. Fox: the profile-dispatch and rectangle_match path, on unlabeled data
uv run lab process Oregon-s_FOX_Autotrader_400 --signal visual
uv run lab doctor Oregon-s_FOX_Autotrader_400

# 5. Proposals from a cross-network audio model
uv run lab propose Oregon-s_FOX_Autotrader_400 --audio-model-from USA_4K_Iowa_Corn_350
# expect: proposal count, anchor coverage %, and how many frames land in `uncertain`

# 6. The UI, against both broadcasts
uv run lab review          # http://localhost:8766/
```

In the browser, confirm on Iowa that every existing ruling, note and facet is
present and the borders/✅❌ marks are unchanged from the current tool; then
switch to the Fox broadcast, filter to `uncertain`, and sweep-confirm a run.

```bash
uv run pytest tests/lab/ -m "not integration"
deno test server/tests/lab/js/
uv run ruff check src/ && uv run ruff format --check src/
```

## Out of scope

Changes to `server/src/tv_commercial_detector/` proper — no audio sensor, no
policy change in the live detector. Those are the recommendations the structure
note left for approval and they stay unimplemented until the lab can measure them
on a held-out broadcast.
