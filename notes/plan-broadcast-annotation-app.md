# Broadcast annotation app: `server/scripts/annotate_broadcasts.py`

## Context

`experiments/review_ground_truth.py` has the review UI that works well for
labeling recorded broadcasts by hand: a cards view with keyboard rulings, a
contact sheet for scanning thousands of thumbnails, a lightbox that walks the
broadcast with a filmstrip and audio, shift-click range selection with a bulk
editor, and URL-addressable state. But it is welded to one experiment: it
hardcodes one broadcast path and three experiment datasets, loads
`truth.json`/`evidence.jsonl`/`replay.json`, and most of its eighteen filters
(`conflict`, `cross_conflict`, `model_wrong`, `contradicts`, ...) only mean
something against AI-generated ground truth from two overlapping captures.

The goal is a new single-file service with the same interaction contract but no
experiment coupling: point it at a `full_broadcasts` root, pick any recording
from a dropdown, and annotate it. The user's decisions for this build:

- **Storage:** `<broadcast>/annotations.json` beside the media, mirroring how
  the live save dir keeps `labels.json` beside its frames.
- **Vocabulary:** binary verdict `ad`/`content`, an `exclude` flag for the
  genuinely undecidable frame (a scoring mask, not a third class), a free-text
  note, and the facet axes from `review_facets.json` (`video`, `audio`, `risk`,
  `care_away`, `care_back`) edited by hand with no inference from notes.
- **Import:** a one-off subcommand that brings the 4775 Iowa Corn 350 rulings
  and facets over from `experiments/review_verdicts.json` / `review_facets.json`.

`notes/plan-reusable-broadcast-lab.md` is to be ignored (the user says its
implementation elsewhere was poor); `experiments/` is not touched.

One finding from surveying the recordings that shapes the design: on YouTube TV
the `video_id` in `page_url` is the *channel*, not the program. `USA_Enjoy_Illinois_300`,
`USA_Law_-_Order-_Special_Victims_Unit`, `USA_USA_Sports_Prerace` and
`USA_USA_Sports_Post-Race` all carry `bLFUzSRlCjg`. So a broadcast is a
recording directory (`<root>/<host>/<dir>/`), identified by that relative path,
and nothing merges on video ID.

## Layout

One new script plus one test module plus a doc section:

```
server/scripts/annotate_broadcasts.py     the service (FastAPI + inline HTML/JS, like record_broadcast.py)
server/tests/test_annotate_broadcasts.py  unit + TestClient tests (imports `from scripts import annotate_broadcasts`)
AGENTS.md                                 "Annotating recorded broadcasts" section under the scripts docs
```

Runtime files, all inside the recording directory (same subdir convention as
`save_dir`: `images/`, `audio/`, `thumbnails/`):

```
<root>/<host>/<dir>/
  classifications.jsonl   manifest written by record_broadcast.py (read only)
  images/  audio/         media (read only)
  thumbnails/             generated on demand by this app (Pillow, 480 px wide JPEG)
  annotations.json        written by this app
```

## CLI

```bash
uv run python scripts/annotate_broadcasts.py                              # serve on :8766
uv run python scripts/annotate_broadcasts.py --root DIR --port 8766 --host 0.0.0.0
uv run python scripts/annotate_broadcasts.py import-verdicts \
    --broadcast tv.youtube.com/USA_4K_Iowa_Corn_350 \
    [--verdicts experiments/review_verdicts.json] [--facets experiments/review_facets.json] \
    [--dataset structure] [--overwrite] [--apply]                          # dry run without --apply
```

`--root` defaults to `/mnt/data/tv-commercial-detector/full_broadcasts` and is
a top-level arg shared by both subcommands; `serve` is the default when no
subcommand is given (argparse subparsers with `required=False` +
`set_defaults(cmd="serve")`). Dry-run-by-default with `--apply` matches
`dedupe_frames.py` and `prune_silent_audio.py`.

## Server design

### Library and frames

- `scan_library(root) -> dict[str, Broadcast]`: every `<root>/<host>/<dir>/classifications.jsonl`.
  Id is `"<host>/<dir>"`. Each `Broadcast` carries `root`, `images_dir`,
  `audio_dir`, `thumbs_dir`, `annotations_path`, and reads `video_title` /
  `network_name` / `page_title` from its first record for the dropdown label.
  The one- and two-frame title-flicker stubs are listed too; they cost nothing.
- `Broadcast.frames()` reads `classifications.jsonl`, sorts by
  `(timestamp, filename)` (ISO strings with a fixed offset sort correctly, and
  the Icelandic capture is genuinely out of order on disk), and assigns `i`.
  Row fields: `i`, `filename`, `t` (`video_offset`), `timestamp`, `has_audio`
  (from one `audio/*.wav` listing, as `attach_audio` does today). Cached per
  broadcast and invalidated on the manifest's `(mtime, size)` so a recording
  still being written picks up new frames on the next request; `POST /api/reload`
  rescans the root as well.
- A `positions` map `filename -> i` per broadcast for O(1) neighbour/fill lookups.

### Annotation store

`annotations.json` is `{"schema": 1, "frames": {filename: record}}` with a flat
record:

```json
{"verdict": "ad", "exclude": true, "note": "...", "video": "spot", "audio": "spot_audio",
 "risk": "safe", "care_away": 3, "care_back": 0, "at": "2026-09-15T18:00:00+00:00"}
```

Every key is optional; a record that ends up with no key other than `at` is
removed. Vocabularies are module constants copied from `experiments/facets.py`
(`VIDEO`, `AUDIO`, `RISK`; care is 0-3), and are handed to the page as
`__SCHEMA__` the way the old tool does. `phase`/`where`/`src`/`review` are not
carried: the first two were derived from the experiment's segment structure and
the last two were provenance for inferred values, and there is no inference here.

Writes go through one `AnnotationStore` per broadcast: in-memory dict, a
`threading.Lock` around every read-modify-write (sync FastAPI endpoints run on
a threadpool, so two requests can overlap), atomic `.tmp` + `Path.replace()`
save, `indent=1, sort_keys=True` so diffs stay readable. Reloaded from disk when
the file's mtime changes underneath (hand edits, or a second instance).

### Routes

| Route | Purpose |
|---|---|
| `GET /` | the page; `__DEFAULT_BROADCAST__`, `__SCHEMA__` substituted |
| `GET /api/broadcasts` | `[{id, label, frames, labeled, first, last}]` for the dropdown |
| `GET /api/frames?b=&filter=&page=&per_page=` | one page of rows with their record attached, plus `counts` |
| `GET /api/sheet?b=&filter=` | whole selection, stripped to `filename, i, verdict, exclude, has_note, fraught` |
| `GET /api/context?b=&filename=&radius=` | neighbours in broadcast order (the `+`→space fallback kept) |
| `POST /api/annotate` | one frame: any subset of the record's fields; `""`/`null` clears a field |
| `POST /api/annotate/bulk` | many frames, every field independently "leave alone" / set / clear; note modes `replace`/`fill`/`append` with `apply_note` ported verbatim |
| `POST /api/fill` | `{b, filename, verdict}`: see "Fill to here" |
| `POST /api/reload` | rescan root, drop frame caches |
| `GET /image/{host}/{dir}/{filename}` | full frame |
| `GET /thumb/{host}/{dir}/{filename}` | 480 px thumbnail, generated into `thumbnails/` on first request |
| `GET /audio/{host}/{dir}/{filename}` | the clip with the same stem |

Media routes resolve `host/dir` against the scanned library (unknown id is a
404, so nothing outside the root is ever served) and reject filenames
containing `/` or `..`.

### Filters

Each is a predicate on `(row, record)`; `counts_for` feeds the stats bar.

| filter | meaning |
|---|---|
| `all` | everything |
| `unlabeled` | no verdict and not excluded, the working set |
| `labeled` | has a verdict |
| `ad` / `content` / `exclude` | by ruling |
| `boundary` | verdict differs from the previous or next frame's in broadcast order; the segment edges, for checking cut placement |
| `noted` | carries a note |
| `unfaceted` | has a verdict but no `video` or no `audio` facet |
| `fraught` | `risk == "fraught"` |
| `ad_read` | `audio == "ad_read"` |

### Fill to here

The lab plan's observation that a broadcast is ~35 boundary decisions rather
than 5000 frame decisions is right, and the old tool already approximates it by
hand with shift-click ranges. `POST /api/fill` makes it one keystroke without
changing the data model: from the frame's position, walk backwards in broadcast
order to the last frame carrying a verdict, and set `verdict` on every
unexcluded frame after it up to and including this one. Returns the filenames
touched and the current frame's previous record so the client can offer undo.
Excluded frames inside the run keep their exclusion and get no verdict.

## Page design

The old `_HTML` is the template for structure and CSS, rewritten rather than
copied so nothing dataset-shaped survives. Everything below the header is the
same interaction contract:

- **Header:** broadcast `<select>` (grouped by host, label `dir (frames, labeled)`),
  filter buttons, reload, counter; stats bar of filter counts.
- **Cards view:** 320 px cards with the thumbnail, filename (selectable without
  moving focus), rows for `i`/offset/timestamp, verdict + exclude badges, the
  facet block (video/audio/risk selects, care 0-3 buttons), audio player, note
  input, and the action buttons. Border colour carries the verdict (red ad, green
  content, grey excluded, none unlabeled). Selection is a shadow, never a border.
- **Contact sheet:** whole selection, thumbnail size slider, border = verdict,
  corner dot = note present, amber dot = fraught; plain click focuses, click on
  the focused tile or `Enter` jumps to its cards page (position ÷ per_page).
- **Lightbox:** full image, meta line, audio, filmstrip of ±12 neighbours in
  broadcast order; `←/→` walk, `space` plays, `s` opens the frame in the sheet
  (widening the filter to `all` once if it isn't in the selection).
- **Range + bulk editor:** shift-click two frames in either view, `e` opens the
  editor; every field has "leave as is" and its own set/clear, note modes with
  the same explanatory hint, nothing written until Save, focus returns to the
  range's second click on close.
- **URL state:** `b`, `filter`, `view`, `page`, `per_page`, `thumb`, `frame`;
  popstate restores.
- **Focus blink** after a view switch, cancelled by the first keystroke;
  one-clip-at-a-time audio via a captured `play` listener; the modifier-key
  guard so browser shortcuts survive; inline handlers named `focusCard`, never
  `select`.

Keys (cards view unless noted):

| key | action |
|---|---|
| `a` / `r` | rule ad / content, advance one card |
| `o` | toggle exclude (the old `other` key; "not one of the two") |
| `x` | clear the ruling and exclusion, leave note and facets |
| `A` / `R` | fill to here as ad / content (also in the sheet and lightbox) |
| `u` | undo the last fill |
| `n` | focus the selected card's note; `Esc` leaves it |
| `j`/`k`/arrows, `Enter`, `s`, shift-click + `e`, `Esc` | as today |
| lightbox: `a`/`r`/`o`/`x` | rule the frame on screen without closing |

After a fill, a transient message under the controls says what was written
(`filled 143 frames as content, i=200…342`) so a one-key sweep is never silent.

## Import subcommand

`import-verdicts` reads the named dataset from `review_verdicts.json` (only
`structure` carries rulings) and `review_facets.json`, and merges into the
target broadcast's `annotations.json`:

| old | new |
|---|---|
| `verdict: ad/content` | `verdict` |
| `verdict: other` | no verdict; note kept, so it lands in `unlabeled` for re-adjudication |
| `note` | `note` |
| facets `video`, `audio`, `risk`, `care_away`, `care_back` (non-null only) | same fields |
| facet `artifact: true` | `exclude: true` |
| `judged`, `at`, `src`, `phase`, `where`, `review`, `confirmed`, `signal_idea` | dropped |

Frames absent from the broadcast's manifest are reported and skipped. Existing
records are left alone unless `--overwrite`. Dry run prints the counts per row
of the table above; `--apply` writes.

## Tests (`tests/test_annotate_broadcasts.py`)

Against a `tmp_path` tree with two fake broadcasts (tiny JPEGs via Pillow, a
couple of `.wav` stems, out-of-order manifest lines):

- library discovery, ids, dropdown labels, frames sorted by timestamp with `i` assigned
- manifest append is picked up without reload (mtime/size invalidation)
- store round trip, empty-record pruning, atomic write leaves no `.tmp`
- `apply_note` modes; bulk with mixed leave/set/clear per field
- fill: stops at the previous verdict, includes the current frame, skips excluded frames, works from `i=0`
- filters, especially `boundary` and `unlabeled`
- import mapping per the table, dry run writes nothing, `--overwrite` semantics
- TestClient: paging, 400 on unknown filter, 404 on unknown broadcast, media path validation, thumbnail generated once

## Verification

```bash
cd server
uv run pytest tests/test_annotate_broadcasts.py
uv run ruff check scripts/annotate_broadcasts.py && uv run ruff format --check scripts/annotate_broadcasts.py

# dry-run then apply the Iowa import, expect ~4666 verdicts, 109 other→unlabeled, 4 artifact→exclude
uv run python scripts/annotate_broadcasts.py import-verdicts --broadcast tv.youtube.com/USA_4K_Iowa_Corn_350
uv run python scripts/annotate_broadcasts.py import-verdicts --broadcast tv.youtube.com/USA_4K_Iowa_Corn_350 --apply

uv run python scripts/annotate_broadcasts.py      # http://localhost:8766/
```

In the browser: the dropdown lists all sixteen recordings; Iowa shows its
imported rulings and facets in cards and sheet; switch to
`Oregon-s_FOX_Autotrader_400`, scan the sheet, `Enter` a frame, `A` to fill a
break, `u` to undo it, shift-click a range and set a facet across it in the
bulk editor; reload the page and confirm the URL restores the same view.
