#!/usr/bin/env python3
"""Standalone FastAPI viewer for auditing the experiments' ground truth.

The numbers in `notes/temporal-hysteresis-2026-08.md` and
`notes/broadcast-structure-2026-08.md` are all measured against labels that were
assigned by reading contact sheets, not by the operator. This app puts each
labelled frame back in front of a human alongside every independent signal that
bears on it, so the labels themselves can be checked.

Nothing here writes to the experiment data. Human rulings go to a separate
`review_verdicts.json`, so the audit and the thing being audited stay distinct.

A ruling records what the frame *is* — `ad`, `content` or `other` — not whether
the stored label was right. Agreement follows from that, and the reverse does
not: "disagree" on an `ad` label never says whether the reviewer meant content
or one of the bumper and sponsor-billboard cases the labelling rule itself
declines to decide. Each ruling also stores the label it was made against, so a
later relabelling cannot quietly convert agreement into dissent.

The captures overlap on 1572 frames, so one frame can be reviewed twice under
two different stored labels. Rulings made elsewhere are shown on the card and
the `contradicts` filter collects any frame ruled two ways.

Usage:
    uv run python experiments/review_ground_truth.py
    uv run python experiments/review_ground_truth.py --port 8766 --dataset structure

Two views. Cards carry every signal for one frame at a time and is where rulings
are made; the contact sheet drops to bordered thumbnails of the whole selection,
for finding the odd ones out by eye. Clicking a thumbnail opens the cards page
holding it, with that frame selected - which works because both views order the
selection identically, so position divided by page size is the page.

Every option is a query param (`dataset`, `filter`, `view`, `page`, `per_page`,
`thumb`, `frame`), so any view is linkable and survives a reload, and back and
forward move between views.

Review order matters more than volume. Three filters carry most of the value:

- `conflict` — an OpenCV anchor contradicts the label. Objectively suspect, and
  small enough to clear in one sitting.
- `cross_conflict` — the two experiments labelled the same frame differently.
  Their captures overlap, so these are two independent passes disagreeing; at
  least one label is wrong.
- `unanchored` — nothing behind the label but the original eyeball pass.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
SERVER = HERE.parent
BROADCAST = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Iowa_Corn_350"
)
VERDICTS = HERE / "review_verdicts.json"

# How near a segment edge a frame has to be to count as a boundary frame. The
# ground truth's edges were placed by eye to +/-30 frames and then refined, so
# this is the window where residual error is plausible.
BOUNDARY_WINDOW = 3


# ── Loading ───────────────────────────────────────────────────────────────────

def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _json(path: Path):
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def hysteresis_labels() -> dict[str, str]:
    """Every frame the hysteresis experiment labelled, keyed by filename.

    Its continuous capture is a prefix of the structure capture, so the same
    frames carry two labels assigned by two independent passes. Where they
    differ, at least one is wrong.
    """
    out: dict[str, str] = {}
    for name in ("cont_dataset.jsonl", "dataset.jsonl"):
        for r in _jsonl(HERE / "hysteresis" / name):
            if r.get("gt"):
                out[r["filename"]] = r["gt"]
    return out


def load_structure() -> list[dict]:
    """The 4775-frame whole-broadcast capture, labelled by segment."""
    truth = _json(HERE / "structure" / "truth.json")
    if not truth:
        return []
    cross = hysteresis_labels()
    visual = _jsonl(HERE / "structure" / "visual.jsonl")
    evidence = {r["i"]: r for r in _jsonl(HERE / "structure" / "evidence.jsonl")}
    furniture = {r["filename"]: r for r in _jsonl(HERE / "structure" / "furniture.jsonl")}
    audio = {r["filename"]: r for r in _jsonl(HERE / "structure" / "audio.jsonl")}

    # Frame index -> the segment covering it, so each frame can show which run
    # of the ground truth put it where it is.
    seg_of: dict[int, dict] = {}
    edges: set[int] = set()
    for s in truth["segments"]:
        for i in range(s["start"], s["end"] + 1):
            seg_of[i] = s
        edges.add(s["start"])
        edges.add(s["end"])

    rows = []
    for v in visual:
        i, fn = v["i"], v["filename"]
        gt = truth["labels"].get(fn)
        if gt is None:
            continue
        ev = evidence.get(i, {})
        seg = seg_of.get(i, {})
        anchor = ev.get("anchor") or ""
        near_edge = min((abs(i - e) for e in edges), default=999)
        rows.append({
            "dataset": "structure",
            "i": i,
            "filename": fn,
            "t": ev.get("t"),
            "gt": gt,
            "anchor": anchor,
            "cross_gt": cross.get(fn),
            "conflict": bool(anchor) and anchor != gt,
            "cross_conflict": bool(cross.get(fn)) and cross[fn] != gt,
            "unanchored": not anchor,
            "boundary": near_edge <= BOUNDARY_WINDOW,
            "boundary_dist": near_edge,
            "seg": f"{seg.get('start')}-{seg.get('end')}" if seg else "",
            "seg_kind": seg.get("kind") or "",
            "bug": ev.get("bug"),
            "banner": ev.get("banner"),
            "black": ev.get("black"),
            "p_ad": ev.get("p_audio+furniture"),
            "p_furniture": ev.get("p_furniture"),
            "p_audio": ev.get("p_audio"),
            "peacock": v.get("peacock"),
            "usa": v.get("usa"),
            "sbs": v.get("sbs"),
            "edge_all": (furniture.get(fn) or {}).get("edge_all"),
            "rms_db": (audio.get(fn) or {}).get("rms_db"),
            "preds": [],
        })
    return rows


def _load_hysteresis(name: str, dataset: str, replay_name: str, images: Path) -> list[dict]:
    rows_in = _jsonl(HERE / "hysteresis" / dataset)
    if not rows_in:
        return []
    replay = _json(HERE / "hysteresis" / replay_name) or {}
    # The continuous capture is a prefix of the structure capture, so the
    # structure ground truth is an independent second opinion on the same frames.
    truth = _json(HERE / "structure" / "truth.json") or {"labels": {}}
    other = truth["labels"]

    rows = []
    for r in rows_in:
        fn = r["filename"]
        preds = replay.get(fn, [])
        cross = other.get(fn)
        rows.append({
            "dataset": name,
            "i": r["i"],
            "filename": fn,
            "t": r.get("video_offset"),
            "timestamp": r.get("timestamp"),
            "gt": r.get("gt"),
            "cross_gt": cross,
            "anchor": "",
            "conflict": False,
            "cross_conflict": bool(cross) and cross != r.get("gt"),
            "unanchored": cross is None,
            "boundary": False,
            "boundary_dist": 999,
            "episode": r.get("episode"),
            "live_class": r.get("live_class"),
            "live_reason": r.get("live_reason"),
            "correct_label": r.get("correct_label"),
            "peacock": r.get("peacock"),
            "usa": r.get("usa"),
            "sbs": r.get("sbs"),
            "images_dir": str(images),
            "preds": preds,
        })
    return rows


def summarize_preds(row: dict) -> None:
    """Fold the replay reps into a majority verdict and a flip flag."""
    preds = row.get("preds") or []
    if not preds:
        row["pred"] = None
        row["pred_agrees"] = None
        row["flips"] = False
        return
    types = [p.get("type") for p in preds]
    top = max(set(types), key=types.count)
    row["pred"] = top
    row["pred_agrees"] = (top == row["gt"])
    row["flips"] = len(set(types)) > 1
    row["pred_reason"] = preds[0].get("reason")


DATASETS: dict[str, list[dict]] = {}
# dataset -> filename -> position in DATASETS[dataset], for O(1) neighbour lookup.
POSITIONS: dict[str, dict[str, int]] = {}


def attach_audio(rows: list[dict]) -> None:
    """Flag which frames have a clip, from one listing rather than a stat per row.

    Clips sit in an `audio/` sibling of `images/` under the same stem. A frame
    without one is normal - the extension only sends audio while the native host
    is connected - so this decides whether to offer a player, not whether
    anything is wrong.
    """
    if not rows:
        return
    audio_dir = Path(rows[0]["images_dir"]).parent / "audio"
    stems = {p.stem for p in audio_dir.glob("*.wav")} if audio_dir.is_dir() else set()
    for r in rows:
        r["audio_dir"] = str(audio_dir)
        r["has_audio"] = Path(r["filename"]).stem in stems


def load_all() -> None:
    structure = load_structure()
    for r in structure:
        r["images_dir"] = str(BROADCAST / "images")
    burst = _load_hysteresis(
        "burst", "dataset.jsonl", "replay.json", SERVER / "frames" / "images"
    )
    cont = _load_hysteresis(
        "cont", "cont_dataset.jsonl", "cont_replay.json", BROADCAST / "images"
    )
    for rows in (structure, burst, cont):
        for r in rows:
            summarize_preds(r)
        attach_audio(rows)
    DATASETS.clear()
    DATASETS.update({"structure": structure, "burst": burst, "cont": cont})
    POSITIONS.clear()
    POSITIONS.update({
        name: {r["filename"]: n for n, r in enumerate(rows)}
        for name, rows in DATASETS.items()
    })


# ── Verdicts ──────────────────────────────────────────────────────────────────

def load_verdicts() -> dict:
    return _json(VERDICTS) or {}


def save_verdicts(v: dict) -> None:
    tmp = VERDICTS.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(v, fh, indent=1, sort_keys=True)
    tmp.replace(VERDICTS)


VERDICT_LABELS = ("ad", "content", "other")


class Verdict(BaseModel):
    """A human ruling on one frame.

    The ruling states what the frame actually is, rather than whether the stored
    label is right, because agreement is derivable from that but not the other
    way round: "disagree" on an `ad` label leaves it open whether the reviewer
    meant content or something outside both. `other` covers the bumpers and
    sponsor billboards the labelling rule itself declines to decide.
    """

    dataset: str
    filename: str
    verdict: str | None = None  # one of VERDICT_LABELS
    note: str | None = None  # None leaves any existing note alone
    clear: bool = False


def ruling_agrees(v: dict | None, gt: str) -> bool | None:
    if not v or not v.get("verdict"):
        return None
    return v["verdict"] == gt


def gt_of(dataset: str, filename: str) -> str | None:
    for r in DATASETS.get(dataset, ()):
        if r["filename"] == filename:
            return r["gt"]
    return None


def rulings_elsewhere(dataset: str, filename: str, store: dict) -> list[dict]:
    """Rulings on this same frame recorded while reviewing a different dataset.

    The captures overlap, so one frame can be reviewed twice under two different
    stored labels. The ruling says what the frame *is*, so the two should match
    whatever dataset they were made from; surfacing them is what lets a reviewer
    notice when they do not.
    """
    out = []
    for name, ds in store.items():
        if name == dataset:
            continue
        rec = ds.get(filename)
        if rec and rec.get("verdict"):
            out.append({"dataset": name, "verdict": rec["verdict"],
                        "judged": rec.get("judged")})
    return out


def annotate_rulings(dataset: str, rows: list[dict], store: dict) -> None:
    """Attach other datasets' rulings, and flag any that contradict this one."""
    mine = store.get(dataset, {})
    others = {n: d for n, d in store.items() if n != dataset}
    if not others:
        for r in rows:
            r["elsewhere"] = []
            r["contradiction"] = False
        return
    for r in rows:
        fn = r["filename"]
        found = [{"dataset": n, "verdict": d[fn]["verdict"], "judged": d[fn].get("judged")}
                 for n, d in others.items() if d.get(fn, {}).get("verdict")]
        r["elsewhere"] = found
        here = (mine.get(fn) or {}).get("verdict")
        r["contradiction"] = bool(here) and any(f["verdict"] != here for f in found)


# ── Filters ───────────────────────────────────────────────────────────────────

FILTERS = {
    "conflict": lambda r, v: r["conflict"],
    "cross_conflict": lambda r, v: r["cross_conflict"],
    "unanchored": lambda r, v: r["unanchored"],
    "boundary": lambda r, v: r["boundary"],
    "model_wrong": lambda r, v: r.get("pred_agrees") is False,
    "model_flips": lambda r, v: r.get("flips"),
    "ad": lambda r, v: r["gt"] == "ad",
    "content": lambda r, v: r["gt"] == "content",
    "reviewed": lambda r, v: v is not None,
    "unreviewed": lambda r, v: v is None,
    # The label was wrong: a ruling was made and it is not what the label says.
    "disputed": lambda r, v: ruling_agrees(v, r["gt"]) is False,
    "confirmed": lambda r, v: ruling_agrees(v, r["gt"]) is True,
    # Same frame ruled two different ways across datasets. Only reachable on the
    # 1572-frame overlap, and always a mistake on the reviewer's part.
    "contradicts": lambda r, v: bool(r.get("contradiction")),
    "all": lambda r, v: True,
}


def app_factory(default_dataset: str) -> FastAPI:
    app = FastAPI(title="Ground truth review")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _HTML.replace("__DEFAULT_DATASET__", json.dumps(default_dataset))

    @app.get("/api/frames")
    def frames(
        dataset: str = "structure",
        filter: str = "conflict",
        page: int = 1,
        per_page: int = 60,
    ) -> JSONResponse:
        rows = DATASETS.get(dataset)
        if rows is None:
            raise HTTPException(404, f"unknown dataset {dataset!r}")
        pred = FILTERS.get(filter)
        if pred is None:
            raise HTTPException(400, f"unknown filter {filter!r}")
        store = load_verdicts()
        verdicts = store.get(dataset, {})
        annotate_rulings(dataset, rows, store)
        sel = [r for r in rows if pred(r, verdicts.get(r["filename"]))]
        total = len(sel)
        per_page = max(1, min(per_page, 500))
        start = (page - 1) * per_page
        page_rows = []
        for r in sel[start:start + per_page]:
            out = dict(r)
            out["verdict"] = verdicts.get(r["filename"])
            page_rows.append(out)
        return JSONResponse({
            "rows": page_rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, (total + per_page - 1) // per_page),
            "counts": counts_for(rows, verdicts),
        })

    @app.get("/api/summary")
    def summary() -> JSONResponse:
        store = load_verdicts()
        out = {}
        for name, rows in DATASETS.items():
            if not rows:
                continue
            annotate_rulings(name, rows, store)
            out[name] = counts_for(rows, store.get(name, {}))
        return JSONResponse(out)

    @app.post("/api/verdict")
    def set_verdict(v: Verdict) -> JSONResponse:
        if v.dataset not in DATASETS:
            raise HTTPException(404, f"unknown dataset {v.dataset!r}")
        if v.verdict is not None and v.verdict not in VERDICT_LABELS:
            raise HTTPException(400, f"verdict must be one of {VERDICT_LABELS}")
        store = load_verdicts()
        ds = store.setdefault(v.dataset, {})
        if v.clear:
            ds.pop(v.filename, None)
        else:
            rec = dict(ds.get(v.filename) or {})
            if v.verdict is not None:
                rec["verdict"] = v.verdict
                # What the experiment claimed at the moment of the ruling, so a
                # later relabelling cannot silently turn agreement into dissent.
                rec["judged"] = gt_of(v.dataset, v.filename)
            if v.note is not None:
                rec["note"] = v.note
            rec["at"] = datetime.now(UTC).isoformat(timespec="seconds")
            ds[v.filename] = rec
        save_verdicts(store)
        return JSONResponse({
            "ok": True,
            "verdict": ds.get(v.filename),
            "elsewhere": rulings_elsewhere(v.dataset, v.filename, store),
        })

    @app.get("/api/sheet")
    def sheet(dataset: str = "structure", filter: str = "conflict") -> JSONResponse:
        """Every frame matching the filter, in capture order, stripped to the bone.

        The contact sheet exists to be scanned in bulk, so it takes the whole
        selection rather than a page - the position in this list is what tells
        the client which page a frame lands on when jumping back to the cards.
        """
        rows = DATASETS.get(dataset)
        if rows is None:
            raise HTTPException(404, f"unknown dataset {dataset!r}")
        pred = FILTERS.get(filter)
        if pred is None:
            raise HTTPException(400, f"unknown filter {filter!r}")
        store = load_verdicts()
        verdicts = store.get(dataset, {})
        annotate_rulings(dataset, rows, store)
        out = []
        for r in rows:
            if not pred(r, verdicts.get(r["filename"])):
                continue
            rec = verdicts.get(r["filename"]) or {}
            out.append({
                "filename": r["filename"], "i": r["i"], "gt": r["gt"],
                "ruling": rec.get("verdict"),
                "conflict": r["conflict"] or r["cross_conflict"],
                "contradiction": r.get("contradiction", False),
                "model_wrong": r.get("pred_agrees") is False,
            })
        return JSONResponse({"rows": out, "total": len(out),
                             "counts": counts_for(rows, verdicts)})

    @app.get("/api/context")
    def context(dataset: str, filename: str, radius: int = 10) -> JSONResponse:
        """The frames either side of this one, in capture order.

        Taken from the whole dataset rather than the current page: the point is
        to see what the broadcast was doing around a frame, and the filtered
        page is by construction a set of scattered oddities.
        """
        rows = DATASETS.get(dataset)
        if rows is None:
            raise HTTPException(404, f"unknown dataset {dataset!r}")
        index = POSITIONS.get(dataset, {})
        pos = index.get(filename)
        if pos is None:
            # Recording filenames carry a `+` for the UTC offset, and a query
            # string decodes `+` as a space. The UI encodes it, but a
            # hand-written URL will not.
            pos = index.get(filename.replace(" ", "+"))
        if pos is None:
            raise HTTPException(404, f"no such frame {filename}")
        radius = max(1, min(radius, 40))
        lo, hi = max(0, pos - radius), min(len(rows), pos + radius + 1)
        store = load_verdicts().get(dataset, {})
        out = []
        for r in rows[lo:hi]:
            rec = store.get(r["filename"]) or {}
            out.append({
                "filename": r["filename"], "i": r["i"], "t": r.get("t"),
                "gt": r["gt"], "anchor": r.get("anchor") or "",
                "pred": r.get("pred"), "has_audio": r.get("has_audio"),
                "seg_kind": r.get("seg_kind") or "",
                "ruling": rec.get("verdict"),
                "current": r["filename"] == filename,
            })
        return JSONResponse({"frames": out, "pos": pos - lo, "total": len(rows)})

    @app.get("/audio/{dataset}/{filename}")
    def audio(dataset: str, filename: str) -> FileResponse:
        rows = DATASETS.get(dataset)
        if not rows:
            raise HTTPException(404, "unknown dataset")
        if "/" in filename or ".." in filename:
            raise HTTPException(400, "bad filename")
        path = Path(rows[0]["audio_dir"]) / (Path(filename).stem + ".wav")
        if not path.exists():
            raise HTTPException(404, f"no clip for {filename}")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/api/reload")
    def reload() -> JSONResponse:
        """Re-read the experiment files. Useful while a replay is still running."""
        load_all()
        return JSONResponse({n: len(r) for n, r in DATASETS.items()})

    @app.get("/image/{dataset}/{filename}")
    def image(dataset: str, filename: str) -> FileResponse:
        rows = DATASETS.get(dataset)
        if not rows:
            raise HTTPException(404, "unknown dataset")
        if "/" in filename or ".." in filename:
            raise HTTPException(400, "bad filename")
        path = Path(rows[0]["images_dir"]) / filename
        if not path.exists():
            raise HTTPException(404, f"no such frame {filename}")
        return FileResponse(path)

    return app


def counts_for(rows: list[dict], verdicts: dict) -> dict[str, int]:
    c = {k: 0 for k in FILTERS}
    for r in rows:
        v = verdicts.get(r["filename"])
        for k, fn in FILTERS.items():
            if fn(r, v):
                c[k] += 1
    return c


# ── HTML ──────────────────────────────────────────────────────────────────────
# Placeholders are substituted at runtime so the JS braces need no escaping.

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ground truth review</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 1rem; }
  h1 { margin: 0 0 0.75rem; font-size: 1.05rem; opacity: 0.6; }
  a { color: #7ab; }

  .bar {
    background: #252525; border-radius: 8px; padding: 0.6rem 1rem;
    display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; align-items: center;
    margin-bottom: 0.75rem;
  }
  .stat { display: flex; flex-direction: column; align-items: center; min-width: 3.5rem; }
  .stat .val { font-size: 1.25rem; font-weight: bold; }
  .stat .lbl { font-size: 0.65rem; opacity: 0.5; text-transform: uppercase; letter-spacing: .03em; }
  .stat.bad .val { color: #f55; }
  .stat.good .val { color: #4d4; }

  #controls { margin-bottom: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.4rem; align-items: center; }
  #controls2 { margin-bottom: 0.6rem; display: flex; gap: 0.75rem; align-items: center; }
  button, select {
    padding: 0.28rem 0.7rem; border: 2px solid #444; border-radius: 4px;
    background: transparent; color: #ccc; cursor: pointer; font-size: 0.8rem;
  }
  button:hover { background: #2e2e2e; }
  button.active { background: #383838; border-color: #777; color: #fff; }
  select { background: #1a1a1a; }
  #counter { margin-left: auto; opacity: 0.5; font-size: 0.8rem; }
  .hint { font-size: 0.72rem; opacity: 0.4; margin-bottom: 0.6rem; }

  .grid { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .card {
    background: #242424; border-radius: 8px; overflow: hidden;
    width: 320px; display: flex; flex-direction: column; border: 2px solid #3a3a3a;
  }
  .card.sel { border-color: #7ab; box-shadow: 0 0 0 2px #7ab4; }
  /* Ruled, and the ruling matches the stored label / contradicts it / neither. */
  .card.confirmed { border-color: #2a5; }
  .card.disputed  { border-color: #c33; }
  .card.other     { border-color: #c92; }
  .card.contradiction { border-color: #d0f; box-shadow: 0 0 0 2px #d0f4; }
  .card img { width: 100%; display: block; cursor: zoom-in; background: #111; min-height: 80px; }
  .fn { padding: 0.25rem 0.5rem; font-size: 0.6rem; opacity: 0.35; word-break: break-all; }
  .rows { padding: 0.35rem 0.5rem; display: flex; flex-direction: column; gap: 0.22rem; font-size: 0.75rem; }
  .row { display: flex; gap: 0.4rem; align-items: baseline; }
  .k { opacity: 0.42; width: 5.2rem; flex-shrink: 0; font-size: 0.7rem; }
  .badge { display: inline-block; padding: 0.08rem 0.4rem; border-radius: 3px; font-size: 0.7rem; font-weight: bold; }
  .badge.ad { background: #a02; color: #fff; }
  .badge.content { background: #060; color: #dfd; }
  .badge.unknown, .badge.none { background: #555; color: #ccc; }
  .warn { color: #f90; }
  audio.clip { width: calc(100% - 1rem); margin: 0.15rem 0.5rem; height: 1.9rem; }
  .reply {
    margin: 0.25rem 0.5rem 0.4rem; padding: 0.35rem 0.45rem; background: #161616;
    border-radius: 4px; border-left: 2px solid #c60; font-size: 0.68rem;
    line-height: 1.45; color: #bbb; max-height: 6rem; overflow-y: auto;
  }
  .acts { display: flex; gap: 0.25rem; padding: 0.35rem 0.5rem 0.5rem; }
  .acts button { flex: 1; padding: 0.25rem 0; font-size: 0.72rem; }
  .note { width: 100%; background: #1a1a1a; color: #ccc; border: 1px solid #444;
          border-radius: 4px; font-size: 0.7rem; padding: 0.2rem 0.35rem; }

  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.95);
              flex-direction: column; align-items: center; justify-content: center;
              z-index: 100; padding: 1rem; gap: .5rem; }
  #lightbox.open { display: flex; }
  #lb-img { max-width: 92vw; max-height: 66vh; border-radius: 6px; }
  #lb-close { position: fixed; top: .5rem; right: 1rem; font-size: 2.4rem; cursor: pointer; color: #fff; }
  #lb-meta { font-size: .8rem; color: #ccc; display: flex; gap: 1rem; align-items: center; }
  #lb-audio { height: 2rem; }

  /* Filmstrip: the frames either side, in capture order. */
  #strip { display: flex; gap: 3px; overflow-x: auto; max-width: 96vw; padding: .3rem 0 .5rem; }
  .sf { flex: 0 0 auto; width: 104px; cursor: pointer; border: 2px solid transparent;
        border-radius: 4px; overflow: hidden; background: #111; position: relative; }
  .sf img { width: 100%; display: block; }
  .sf .cap { font-size: .58rem; text-align: center; padding: 1px 0; color: #ccc; }
  .sf.ad      { border-color: #a02; }
  .sf.content { border-color: #060; }
  .sf.cur     { border-color: #7ab; box-shadow: 0 0 0 2px #7ab; }
  .sf .mark { position: absolute; top: 1px; right: 2px; font-size: .6rem;
              background: #000a; border-radius: 2px; padding: 0 2px; }

  /* Contact sheet: the whole selection at a glance, for picking out oddities.
     Border carries the stored label, the corner dot carries a ruling. */
  #sheet { display: flex; flex-wrap: wrap; gap: 2px; }
  .th { position: relative; cursor: pointer; border: 2px solid #333;
        border-radius: 3px; overflow: hidden; background: #111; line-height: 0; }
  .th img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .th.ad      { border-color: #a02; }
  .th.content { border-color: #060; }
  .th.conflict      { outline: 1px solid #fa0; outline-offset: -3px; }
  .th.contradiction { outline: 2px solid #d0f; outline-offset: -4px; }
  .th.focus { border-color: #7ab; box-shadow: 0 0 0 2px #7ab; }
  .th .dot { position: absolute; top: 2px; right: 2px; width: 7px; height: 7px;
             border-radius: 50%; border: 1px solid #000a; }
  .th .dot.ad { background: #f44; } .th .dot.content { background: #4d4; }
  .th .dot.other { background: #fb3; }
  .th .idx { position: absolute; bottom: 0; left: 0; right: 0; font-size: .5rem;
             text-align: center; background: #000a; color: #ddd; line-height: 1.3; }
  #size { width: 8rem; }
  .legend { font-size: .68rem; opacity: .45; display: flex; gap: .9rem; flex-wrap: wrap;
            margin-bottom: .5rem; }
</style>
</head>
<body>
<h1>Ground truth review — is the label right?</h1>
<div class="bar" id="bar"></div>
<div id="controls">
  <select id="dataset">
    <option value="structure">structure (4775, whole broadcast)</option>
    <option value="burst">burst (1400, 2026-08-09)</option>
    <option value="cont">cont (1572, continuous)</option>
  </select>
  <button data-f="conflict">Anchor conflicts</button>
  <button data-f="cross_conflict">Cross-experiment conflicts</button>
  <button data-f="unanchored">Unanchored</button>
  <button data-f="boundary">Boundaries</button>
  <button data-f="model_wrong">Model disagrees</button>
  <button data-f="model_flips">Model flips</button>
  <button data-f="ad">GT ad</button>
  <button data-f="content">GT content</button>
  <button data-f="disputed">Label wrong</button>
  <button data-f="confirmed">Label confirmed</button>
  <button data-f="contradicts">My contradictions</button>
  <button data-f="unreviewed">Unreviewed</button>
  <button data-f="all">All</button>
  <button id="reload" title="Re-read the experiment files">⟳</button>
  <span id="counter"></span>
</div>
<div id="controls2">
  <button id="viewtoggle" title="Switch between review cards and a scannable contact sheet"></button>
  <label id="sizewrap" style="font-size:.75rem;opacity:.6;display:none">
    thumb <input type="range" id="size" min="48" max="240" step="8" value="112">
  </label>
</div>
<div class="hint">
  Say what the frame <i>is</i>, not whether the label is right — agreement is worked out from that.
  <b>a</b> ad · <b>r</b> content/racing · <b>o</b> other (bumper, sponsor billboard, undecidable) ·
  <b>x</b> clear · <b>j/k</b> or arrows to move · <b>Enter</b> or click a frame to open it in context.
  In context view, <b>←/→</b> walk the broadcast, <b>space</b> plays the clip and <b>s</b> shows the frame in the contact sheet.
  Rulings save to <code>experiments/review_verdicts.json</code>.
</div>
<div class="legend" id="legend" style="display:none">
  <span>border: <b style="color:#f55">red</b> label ad · <b style="color:#4d4">green</b> label content</span>
  <span>dot: your ruling (red ad, green content, amber other)</span>
  <span><b style="color:#fa0">amber outline</b> conflicts with a signal or the other pass</span>
  <span><b style="color:#d0f">magenta</b> ruled two ways</span>
  <span>click a thumbnail to open its page in the cards view</span>
</div>
<div class="grid" id="grid"></div>
<div id="sheet" style="display:none"></div>
<div id="pager" style="margin-top:1rem;display:flex;gap:.5rem;align-items:center"></div>
<div id="lightbox">
  <span id="lb-close" title="Close (Esc)">&times;</span>
  <img id="lb-img" src="" alt="">
  <div id="lb-meta"></div>
  <audio id="lb-audio" controls preload="none"></audio>
  <div id="strip"></div>
</div>

<script>
let DATASET = __DEFAULT_DATASET__;
let FILTER = "conflict";
let PAGE = 1;
let VIEW = "cards";      // "cards" | "sheet"
let PER_PAGE = 60;
let THUMB = 112;
let FOCUS = null;        // filename to highlight after a jump from the sheet
let ROWS = [];           // the current cards page
let SHEET = [];          // the whole filtered selection, contact-sheet view
let NOTICE = "";         // one-shot message shown under the sheet
let SEL = 0;

// ── URL state ───────────────────────────────────────────────────────────────
// Every option lives in the query string, so any view is linkable and survives
// a reload - the same contract /review keeps.

function readUrl() {
  const q = new URLSearchParams(location.search);
  DATASET = q.get("dataset") || DATASET;
  FILTER = q.get("filter") || FILTER;
  PAGE = Math.max(1, parseInt(q.get("page") || "1", 10) || 1);
  VIEW = q.get("view") === "sheet" ? "sheet" : "cards";
  PER_PAGE = Math.min(500, Math.max(1, parseInt(q.get("per_page") || "60", 10) || 60));
  THUMB = Math.min(240, Math.max(48, parseInt(q.get("thumb") || "112", 10) || 112));
  FOCUS = q.get("frame");
}

function syncUrl(push) {
  const q = new URLSearchParams();
  q.set("dataset", DATASET);
  q.set("filter", FILTER);
  q.set("view", VIEW);
  if (VIEW === "cards") { q.set("page", PAGE); q.set("per_page", PER_PAGE); }
  else q.set("thumb", THUMB);
  if (FOCUS) q.set("frame", FOCUS);
  const url = location.pathname + "?" + q.toString();
  if (push) history.pushState(null, "", url); else history.replaceState(null, "", url);
}

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const num = (x, d=3) => (x === null || x === undefined) ? "–" : (typeof x === "number" ? x.toFixed(d) : x);

function badge(v) {
  const c = v === "ad" ? "ad" : v === "content" ? "content" : "none";
  return `<span class="badge ${c}">${esc(v ?? "—")}</span>`;
}

async function load() {
  syncUrl(false);
  $("#viewtoggle").textContent = VIEW === "sheet" ? "▤ Cards view" : "▦ Contact sheet";
  $("#sizewrap").style.display = VIEW === "sheet" ? "" : "none";
  $("#legend").style.display = VIEW === "sheet" ? "" : "none";
  $("#grid").style.display = VIEW === "sheet" ? "none" : "";
  $("#sheet").style.display = VIEW === "sheet" ? "" : "none";
  if (VIEW === "sheet") return loadSheet();

  const r = await fetch(`/api/frames?dataset=${DATASET}&filter=${FILTER}&page=${PAGE}&per_page=${PER_PAGE}`);
  if (!r.ok) { $("#grid").innerHTML = `<p class="warn">${esc(await r.text())}</p>`; return; }
  const d = await r.json();
  ROWS = d.rows;
  // A jump from the sheet names a frame; land on it rather than the top.
  const at = FOCUS ? ROWS.findIndex(x => x.filename === FOCUS) : -1;
  SEL = at >= 0 ? at : 0;
  // Arriving with no frame named, the top card is where the view is, so the
  // sheet has somewhere to scroll to if the toggle is hit straight away.
  if (ROWS[SEL]) { FOCUS = ROWS[SEL].filename; syncUrl(false); }
  render(d);
  if (at >= 0) document.querySelectorAll(".card")[at]
    ?.scrollIntoView({ block: "center", behavior: "smooth" });
}

async function loadSheet() {
  const r = await fetch(`/api/sheet?dataset=${DATASET}&filter=${FILTER}`);
  if (!r.ok) { $("#sheet").innerHTML = `<p class="warn">${esc(await r.text())}</p>`; return; }
  const d = await r.json();
  // Arriving from the filmstrip, the frame may sit outside the current filter -
  // the strip walks the whole broadcast. Widen to `all` once so the jump lands
  // on the frame instead of silently on nothing.
  if (FOCUS && FILTER !== "all" && !d.rows.some(n => n.filename === FOCUS)) {
    FILTER = "all";
    syncUrl(false);
    NOTICE = "that frame is outside the previous filter — showing all";
    return loadSheet();
  }
  SHEET = d.rows;
  stats(d.counts);
  document.querySelectorAll("#controls button[data-f]").forEach(b =>
    b.classList.toggle("active", b.dataset.f === FILTER));
  $("#counter").textContent = `${d.total} frames`;
  $("#pager").innerHTML = NOTICE ? `<span class="warn">${esc(NOTICE)}</span>` : "";
  NOTICE = "";
  const w = THUMB, h = Math.round(THUMB * 9 / 16);
  $("#sheet").innerHTML = SHEET.map((n, k) => `
    <div class="th ${n.gt} ${n.conflict ? "conflict" : ""} ${n.contradiction ? "contradiction" : ""} ${n.filename === FOCUS ? "focus" : ""}"
         style="width:${w}px;height:${h}px" onclick="jumpTo(${k})"
         title="i=${n.i} · label ${esc(n.gt)}${n.ruling ? " · ruled " + esc(n.ruling) : ""}">
      <img src="/image/${DATASET}/${encodeURIComponent(n.filename)}" loading="lazy">
      ${n.ruling ? `<span class="dot ${n.ruling}"></span>` : ""}
      <span class="idx">${n.i}</span>
    </div>`).join("");
  if (FOCUS) document.querySelector(".th.focus")
    ?.scrollIntoView({ block: "center", behavior: "smooth" });
}

// Position within the filtered selection decides which cards page holds it.
function jumpTo(k) {
  const n = SHEET[k];
  if (!n) return;
  FOCUS = n.filename;
  PAGE = Math.floor(k / PER_PAGE) + 1;
  VIEW = "cards";
  syncUrl(true);
  load();
}

function stats(c) {
  $("#bar").innerHTML = [
    ["conflict", "anchor", "bad"], ["cross_conflict", "cross-pass", "bad"],
    ["unanchored", "unanchored", ""],
    ["boundary", "boundary", ""], ["model_wrong", "model wrong", ""],
    ["reviewed", "ruled", "good"], ["confirmed", "confirmed", "good"],
    ["disputed", "label wrong", "bad"], ["contradicts", "contradictions", "bad"],
    ["all", "frames", ""],
  ].map(([k, lbl, cls]) => `<div class="stat ${cls}"><span class="val">${c[k] ?? 0}</span><span class="lbl">${lbl}</span></div>`).join("");
}

function render(d) {
  stats(d.counts);
  document.querySelectorAll("#controls button[data-f]").forEach(b =>
    b.classList.toggle("active", b.dataset.f === FILTER));
  $("#counter").textContent = `${d.total} frames · page ${d.page}/${d.pages}`;

  $("#grid").innerHTML = ROWS.map((r, idx) => card(r, idx)).join("");
  $("#pager").innerHTML = d.pages > 1
    ? `<button onclick="go(${Math.max(1, d.page - 1)})">← prev</button>
       <span style="opacity:.5;font-size:.8rem">page ${d.page} of ${d.pages}</span>
       <button onclick="go(${Math.min(d.pages, d.page + 1)})">next →</button>` : "";
  paint();
}

// The stored label is what we are judging; a ruling that matches it confirms
// it, one that differs disputes it, and "other" rejects the binary outright.
function cardClass(r) {
  const v = r.verdict?.verdict;
  if (!v) return "";
  const base = v === "other" ? "other" : (v === r.gt ? "confirmed" : "disputed");
  return r.contradiction ? base + " contradiction" : base;
}

function card(r, idx) {
  const v = r.verdict?.verdict ?? "";
  const rows = [];
  rows.push(`<div class="row"><span class="k">label (AI)</span>${badge(r.gt)} <span style="opacity:.4;font-size:.68rem">${esc(r.seg_kind || "")}</span></div>`);
  if (r.anchor !== undefined && r.anchor !== "")
    rows.push(`<div class="row"><span class="k">opencv anchor</span>${badge(r.anchor)} ${r.conflict ? '<span class="warn">← conflicts</span>' : ""}</div>`);
  if (r.cross_gt)
    rows.push(`<div class="row"><span class="k">other pass</span>${badge(r.cross_gt)} ${r.cross_conflict ? '<span class="warn">← conflicts</span>' : ""}</div>`);
  if (r.pred)
    rows.push(`<div class="row"><span class="k">model</span>${badge(r.pred)} <span style="opacity:.45;font-size:.68rem">${esc(r.pred_reason || "")}${r.flips ? " · flips" : ""}</span></div>`);
  if (r.live_class)
    rows.push(`<div class="row"><span class="k">live said</span>${badge(r.live_class)} <span style="opacity:.45;font-size:.68rem">${esc(r.live_reason || "")}</span></div>`);
  rows.push(`<div class="row"><span class="k">frame</span><span style="opacity:.6">i=${r.i} t=${num(r.t, 1)}s ${r.seg ? "seg " + esc(r.seg) : ""}${r.boundary ? ` <span class="warn">edge±${r.boundary_dist}</span>` : ""}</span></div>`);
  rows.push(`<div class="row"><span class="k">scores</span><span style="opacity:.6">usa ${num(r.usa)} · pea ${num(r.peacock)} · sbs ${num(r.sbs)}</span></div>`);
  if (r.p_ad !== undefined && r.p_ad !== null)
    rows.push(`<div class="row"><span class="k">p(ad)</span><span style="opacity:.6">${num(r.p_ad, 4)} (furn ${num(r.p_furniture)})</span></div>`);

  if (v) {
    const verdictOf = v === "other" ? "neither" : (v === r.gt ? "confirms label" : "label wrong");
    rows.push(`<div class="row"><span class="k">my ruling</span>${badge(v)} <span style="opacity:.5;font-size:.68rem">${verdictOf}</span></div>`);
  }
  for (const e of (r.elsewhere || [])) {
    const clash = v && e.verdict !== v;
    rows.push(`<div class="row"><span class="k">ruled in ${esc(e.dataset)}</span>${badge(e.verdict)}` +
      `${clash ? ' <span class="warn">← contradicts this ruling</span>'
               : ` <span style="opacity:.45;font-size:.68rem">judging ${esc(e.judged ?? "?")}</span>`}</div>`);
  }

  const reply = (r.preds || []).map(p => p.reply).find(x => x && x !== "(opencv)");
  return `<div class="card ${cardClass(r)}" data-idx="${idx}" onclick="select(${idx})">
    <img src="/image/${DATASET}/${encodeURIComponent(r.filename)}" loading="lazy"
         title="Click to see this frame in context"
         onclick="event.stopPropagation();zoom('${r.filename}')">
    <div class="fn">${esc(r.filename)}</div>
    <div class="rows">${rows.join("")}</div>
    ${r.has_audio ? `<audio class="clip" controls preload="none"
         onclick="event.stopPropagation()"
         src="/audio/${DATASET}/${encodeURIComponent(r.filename)}"></audio>` : ""}
    ${reply ? `<div class="reply">${esc(reply)}</div>` : ""}
    <div class="acts">
      <button onclick="event.stopPropagation();mark(${idx},'ad')">ad</button>
      <button onclick="event.stopPropagation();mark(${idx},'content')">content</button>
      <button onclick="event.stopPropagation();mark(${idx},'other')">other</button>
      <button onclick="event.stopPropagation();mark(${idx},null)">×</button>
    </div>
    <div style="padding:0 .5rem .5rem"><input class="note" placeholder="note"
      value="${esc(r.verdict?.note ?? "")}" onclick="event.stopPropagation()"
      onchange="note(${idx}, this.value)"></div>
  </div>`;
}

function paint() {
  document.querySelectorAll(".card").forEach((el, i) => el.classList.toggle("sel", i === SEL));
  const el = document.querySelectorAll(".card")[SEL];
  if (el) el.scrollIntoView({ block: "nearest" });
}

// The selected card is the frame the view is "on", so it is what the contact
// sheet scrolls to and what the URL names. Keeping FOCUS in step here is what
// makes the view toggle carry the current card rather than the last jump.
function setSel(i) {
  SEL = Math.max(0, Math.min(i, ROWS.length - 1));
  const r = ROWS[SEL];
  if (r) { FOCUS = r.filename; syncUrl(false); }
  paint();
}

function select(i) { setSel(i); }
function go(p) { PAGE = p; FOCUS = null; syncUrl(true); load(); }

// ── Lightbox: one frame in the context of its neighbours ────────────────────
let CTX = [];      // the neighbouring frames, in capture order
let CTX_POS = 0;   // which of them is on screen

async function zoom(filename) {
  const r = await fetch(`/api/context?dataset=${DATASET}&filename=${encodeURIComponent(filename)}&radius=12`);
  if (!r.ok) return;
  const d = await r.json();
  CTX = d.frames; CTX_POS = d.pos;
  $("#lightbox").classList.add("open");
  showCtx();
}

function showCtx() {
  const f = CTX[CTX_POS];
  if (!f) return;
  $("#lb-img").src = `/image/${DATASET}/${encodeURIComponent(f.filename)}`;
  const bits = [`i=${f.i}`, f.t != null ? `t=${f.t.toFixed(1)}s` : "",
                `label ${f.gt}`, f.anchor ? `anchor ${f.anchor}` : "",
                f.pred ? `model ${f.pred}` : "", f.ruling ? `ruled ${f.ruling}` : "",
                f.seg_kind].filter(Boolean);
  $("#lb-meta").innerHTML = bits.map(esc).join(" &middot; ") +
    ` <button id="lb-sheet" onclick="openInSheet()" title="Show this frame in the contact sheet (s)">▦ in sheet</button>` +
    `<span style="opacity:.45">&nbsp;←/→ step, <b>s</b> sheet, Esc close</span>`;

  const au = $("#lb-audio");
  if (f.has_audio) {
    au.style.display = "";
    au.src = `/audio/${DATASET}/${encodeURIComponent(f.filename)}`;
  } else {
    au.style.display = "none";
    au.removeAttribute("src");
  }

  $("#strip").innerHTML = CTX.map((n, k) => `
    <div class="sf ${n.gt} ${k === CTX_POS ? "cur" : ""}" onclick="stepTo(${k})">
      <img src="/image/${DATASET}/${encodeURIComponent(n.filename)}" loading="lazy">
      ${n.ruling ? `<span class="mark">${esc(n.ruling[0].toUpperCase())}</span>` : ""}
      <div class="cap">${n.i}</div>
    </div>`).join("");
  const cur = $("#strip").children[CTX_POS];
  if (cur) cur.scrollIntoView({ block: "nearest", inline: "center" });
}

function stepTo(k) { CTX_POS = Math.max(0, Math.min(k, CTX.length - 1)); showCtx(); }

// Jump from the frame on screen straight to its place in the contact sheet.
// The filmstrip walks the whole broadcast while the sheet shows only the
// filtered selection, so the frame may not be in it; loadSheet widens the
// filter rather than landing nowhere.
function openInSheet() {
  const f = CTX[CTX_POS];
  if (!f) return;
  FOCUS = f.filename;
  closeLightbox();
  VIEW = "sheet";
  syncUrl(true);
  load();
}

function closeLightbox() {
  $("#lightbox").classList.remove("open");
  const au = $("#lb-audio");
  au.pause();
  au.removeAttribute("src");
}

async function mark(idx, verdict) {
  const r = ROWS[idx];
  const res = await fetch("/api/verdict", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: DATASET, filename: r.filename, verdict,
                           clear: verdict === null }),
  });
  const d = await res.json();
  r.verdict = d.verdict;
  r.elsewhere = d.elsewhere ?? r.elsewhere ?? [];
  r.contradiction = !!(d.verdict?.verdict &&
    r.elsewhere.some(e => e.verdict !== d.verdict.verdict));
  // Re-render just this card so the ruling row and any contradiction show up.
  const el = document.querySelectorAll(".card")[idx];
  el.outerHTML = card(r, idx);
  paint();
}

// Note only — leaves any existing ruling untouched.
async function note(idx, text) {
  const r = ROWS[idx];
  await fetch("/api/verdict", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset: DATASET, filename: r.filename, note: text }),
  });
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  // Never shadow a browser shortcut: cmd-c / ctrl-a and friends must still work.
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  // While the lightbox is open the arrows walk the broadcast, not the grid.
  if ($("#lightbox").classList.contains("open")) {
    if (e.key === "Escape") return closeLightbox();
    if (e.key === "ArrowLeft"  || e.key === "k") { e.preventDefault(); return stepTo(CTX_POS - 1); }
    if (e.key === "ArrowRight" || e.key === "j") { e.preventDefault(); return stepTo(CTX_POS + 1); }
    if (e.key === "s") { e.preventDefault(); return openInSheet(); }
    if (e.key === " ") {  // play/pause the clip for the frame on screen
      e.preventDefault();
      const au = $("#lb-audio");
      if (au.src) au.paused ? au.play() : au.pause();
      return;
    }
    return;
  }
  if (VIEW !== "cards") return;
  // `r` for racing, not `c`: `c` collided with cmd-c. The modifier guard above
  // now stops that anyway, but the binding is not worth reclaiming.
  const keys = { a: "ad", r: "content", o: "other" };
  if (keys[e.key]) { mark(SEL, keys[e.key]); setSel(SEL + 1); }
  else if (e.key === "x") mark(SEL, null);
  else if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") setSel(SEL + 1);
  else if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") setSel(SEL - 1);
  else if (e.key === "Enter") { const r = ROWS[SEL]; if (r) zoom(r.filename); }
});

// Only the backdrop closes — clicks on the image, strip or player must not.
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") closeLightbox(); };
$("#lb-close").onclick = closeLightbox;
$("#dataset").onchange = (e) => {
  DATASET = e.target.value; PAGE = 1; FOCUS = null; syncUrl(true); load();
};
document.querySelectorAll("#controls button[data-f]").forEach(b =>
  b.onclick = () => { FILTER = b.dataset.f; PAGE = 1; FOCUS = null; syncUrl(true); load(); });
$("#reload").onclick = async () => { await fetch("/api/reload", { method: "POST" }); load(); };
$("#viewtoggle").onclick = () => {
  VIEW = VIEW === "sheet" ? "cards" : "sheet";
  syncUrl(true); load();
};
$("#size").oninput = (e) => {
  THUMB = +e.target.value;
  const w = THUMB, h = Math.round(THUMB * 9 / 16);
  document.querySelectorAll(".th").forEach(t => { t.style.width = w + "px"; t.style.height = h + "px"; });
  syncUrl(false);
};

// Back/forward should restore the view the URL describes, not refetch blindly.
window.addEventListener("popstate", () => {
  readUrl();
  $("#dataset").value = DATASET;
  $("#size").value = THUMB;
  load();
});

readUrl();
$("#dataset").value = DATASET;
$("#size").value = THUMB;
load();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--dataset", default="structure", choices=["structure", "burst", "cont"])
    args = ap.parse_args()

    load_all()
    for name, rows in DATASETS.items():
        n_pred = sum(1 for r in rows if r.get("pred"))
        print(f"{name:10s} {len(rows):5d} frames  {n_pred:5d} with model predictions",
              file=sys.stderr)
    if not any(DATASETS.values()):
        sys.exit("no experiment data found — run this from the server/ checkout")

    print(f"\n  http://localhost:{args.port}/\n", file=sys.stderr)
    uvicorn.run(app_factory(args.dataset), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
