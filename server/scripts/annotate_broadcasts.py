#!/usr/bin/env python3
"""Standalone FastAPI app for annotating recorded broadcasts by hand.

Points at a `record_broadcast.py` output root and lets the operator pick any
recording under it, then rule on its frames one at a time or in runs. Nothing
here knows about a classifier, an experiment or a ground truth: the only
inputs are the recording's `classifications.jsonl` manifest and its media, and
the only output is an `annotations.json` written beside them.

    uv run python scripts/annotate_broadcasts.py
    uv run python scripts/annotate_broadcasts.py --root DIR --port 8766
    uv run python scripts/annotate_broadcasts.py import-verdicts \
        --broadcast tv.youtube.com/USA_4K_Iowa_Corn_350 --apply

A broadcast is a recording directory, `<root>/<host>/<dir>/`, and is identified
by that relative path. It is deliberately not the `video_id` in the manifest:
on YouTube TV that is the channel, not the program, and four different USA
recordings share one.

Each frame carries a binary verdict (`ad` / `content`), an `exclude` flag for
the genuinely undecidable frame (a scoring mask rather than a third class), a
free-text note, and the facet axes describing what is on screen, in the sound,
and what being wrong costs in each direction. All of it is set by hand; nothing
is inferred.

Two views. Cards show one frame at a time with every field editable and is
where rulings are made; the contact sheet drops to bordered thumbnails of the
whole selection, for finding segment edges by eye. Clicking a thumbnail opens
the cards page holding it, which works because both views order the selection
identically. Shift+click two frames in either view and `e` opens a bulk editor
where every field is independently left alone, set or cleared. `A` / `R`
fill from the previous ruled frame through the current one, which is how a
whole segment is ruled in one keystroke; `u` takes it back.

Every option is a query param (`b`, `filter`, `view`, `page`, `per_page`,
`thumb`, `frame`), so any view is linkable and survives a reload.
"""

import argparse
import json
import sys
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent
DEFAULT_ROOT = Path("/mnt/data/tv-commercial-detector/full_broadcasts")
DEFAULT_PORT = 8766
DEFAULT_VERDICTS = SERVER / "experiments" / "review_verdicts.json"
DEFAULT_FACETS = SERVER / "experiments" / "review_facets.json"

MANIFEST = "classifications.jsonl"
ANNOTATIONS = "annotations.json"
SCHEMA_VERSION = 1
THUMB_WIDTH = 480
# Longest run a single fill keystroke may rule. A whole unlabeled broadcast
# would otherwise be one accidental `A` away; a run longer than this is what
# the range editor is for.
FILL_MAX = 900

# ── Vocabularies ──────────────────────────────────────────────────────────────
# The facet axes describe the frame; the verdict decides what the matrix does.
# Values were written down by the operator during the Iowa review, not
# invented as categories that seemed worth having.

VERDICTS = ("ad", "content")

VIDEO = {
    "live_race": "full-screen racing, including in-car and onboard",
    "inset": "racing in a box beside something else - interview, replay, studio",
    "side_by_side": "NASCAR NON STOP: race inset, commercial alongside",
    "bumper": "full-screen network transition banner or wipe",
    "package": "produced, edited, not live, about this event - show opens, "
    "driver features, hype pieces, victory-lane recaps",
    "promo": "produced by the network to sell other programming",
    "spot": "full-screen commercial for somebody else's product",
    "studio": "booth, desk, presenters, no racing on screen",
    "pit": "pit road, paddock, garage, driver interview",
    "crowd": "fans, grandstands, aerials of the venue",
    "black": "blank or near-black frame",
}

AUDIO = {
    "commentary": "the race call",
    "ad_read": "talent reading scripted sponsor copy",
    "spot_audio": "a commercial's own audio",
    "chatter": "non-scripted filler - 'welcome back', track talk",
    "music_bed": "bumper or transition music",
    "silence": "no significant audio",
}

RISK = {
    "safe": "fine to train on",
    "fraught": "correct, but likely to generalise badly - see the note",
}

CARE_MAX = 3

SCHEMA = {
    "video": VIDEO,
    "audio": AUDIO,
    "risk": RISK,
    "care": f"0 don't care … {CARE_MAX} urgent",
}

# Every field a record may carry besides `at`. A patch may name any subset;
# `null` clears the field and an absent key leaves it alone.
FIELDS = (
    "verdict",
    "exclude",
    "note",
    "video",
    "audio",
    "risk",
    "care_away",
    "care_back",
)
VOCAB = {"video": VIDEO, "audio": AUDIO, "risk": RISK}

# How a swept note meets a note already on the frame. Sweeping a run usually
# means recording something about the run as a whole, which should not cost the
# per-frame observations already written there - hence `fill` and `append`.
NOTE_MODES = ("replace", "fill", "append")
NOTE_JOIN = " || "


def apply_note(existing: str, incoming: str, mode: str) -> str:
    if mode == "replace":
        return incoming
    if not incoming:  # nothing to fill or append with
        return existing
    if mode == "fill":
        return existing or incoming
    return f"{existing}{NOTE_JOIN}{incoming}" if existing else incoming


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ── Records ───────────────────────────────────────────────────────────────────


def validate_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Check a patch's keys and values; a bad one is a 400, not a stored oddity."""
    out: dict[str, Any] = {}
    for key, value in patch.items():
        if key not in FIELDS:
            raise HTTPException(400, f"unknown field {key!r}")
        if value is None or value == "":
            out[key] = None
            continue
        if key == "verdict":
            if value not in VERDICTS:
                raise HTTPException(400, f"verdict must be one of {VERDICTS}")
        elif key == "exclude":
            if not isinstance(value, bool):
                raise HTTPException(400, "exclude must be a boolean")
            if not value:
                value = None
        elif key == "note":
            if not isinstance(value, str):
                raise HTTPException(400, "note must be a string")
        elif key in VOCAB:
            if value not in VOCAB[key]:
                raise HTTPException(400, f"{key} must be one of {sorted(VOCAB[key])}")
        else:  # care_away / care_back
            if isinstance(value, bool) or not isinstance(value, int):
                raise HTTPException(400, f"{key} must be an integer")
            if not 0 <= value <= CARE_MAX:
                raise HTTPException(400, f"{key} must be 0-{CARE_MAX}")
        out[key] = value
    return out


def apply_patch(rec: dict, patch: dict[str, Any], note_mode: str = "replace") -> dict:
    """A new record with the patch applied. `None` removes a field."""
    out = dict(rec)
    for key, value in patch.items():
        if key == "note" and value is not None:
            value = apply_note(out.get("note", ""), value, note_mode) or None
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def is_empty(rec: dict) -> bool:
    """A record carrying nothing but its timestamp says nothing; drop it."""
    return not any(k in rec for k in FIELDS)


def summarize(rec: dict | None) -> dict:
    """The few fields the sheet and filmstrip paint, without the whole record."""
    rec = rec or {}
    return {
        "verdict": rec.get("verdict"),
        "exclude": bool(rec.get("exclude")),
        "has_note": bool(rec.get("note")),
        "fraught": rec.get("risk") == "fraught",
    }


# ── Store ─────────────────────────────────────────────────────────────────────


class AnnotationStore:
    """`annotations.json` for one broadcast: `{"schema": 1, "frames": {...}}`.

    In-memory, written through on every change. The lock covers the whole
    read-modify-write because sync endpoints run on a threadpool and two
    requests can overlap; the mtime check picks up a hand edit or a second
    instance writing the same file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self._frames: dict[str, dict] = {}
        self._stamp: tuple[int, int] | None = None

    def _stat(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _refresh(self) -> None:
        stamp = self._stat()
        if stamp == self._stamp:
            return
        if stamp is None:
            self._frames = {}
        else:
            with self.path.open() as fh:
                data = json.load(fh)
            self._frames = data.get("frames", {})
        self._stamp = stamp

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(
                {"schema": SCHEMA_VERSION, "frames": self._frames},
                fh,
                indent=1,
                sort_keys=True,
            )
        tmp.replace(self.path)
        self._stamp = self._stat()

    def all(self) -> dict[str, dict]:
        with self.lock:
            self._refresh()
            return dict(self._frames)

    def get(self, filename: str) -> dict | None:
        with self.lock:
            self._refresh()
            rec = self._frames.get(filename)
            return dict(rec) if rec else None

    def mutate(self, fn: Callable[[dict[str, dict]], Any]) -> Any:
        """Run `fn` on the live mapping under the lock, then save."""
        with self.lock:
            self._refresh()
            result = fn(self._frames)
            self._save()
            return result

    def put(self, filename: str, rec: dict) -> dict | None:
        """Store one record, or drop it if empty. Returns what is stored."""

        def go(frames: dict[str, dict]) -> dict | None:
            if is_empty(rec):
                frames.pop(filename, None)
                return None
            rec["at"] = now()
            frames[filename] = rec
            return rec

        return self.mutate(go)


# ── Broadcasts ────────────────────────────────────────────────────────────────


@dataclass
class Broadcast:
    """One recording directory: manifest, media and the annotations beside them."""

    id: str
    root: Path
    title: str = ""
    network: str = ""
    store: AnnotationStore = field(init=False)
    _frames: list[dict] = field(default_factory=list, init=False, repr=False)
    _positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _stamp: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.store = AnnotationStore(self.root / ANNOTATIONS)

    @property
    def manifest(self) -> Path:
        return self.root / MANIFEST

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def thumbs_dir(self) -> Path:
        return self.root / "thumbnails"

    def frames(self) -> list[dict]:
        """Every frame in broadcast order, reloaded when the manifest grows.

        A recording still being written appends to the manifest, so the stamp
        check is what makes new frames appear without a restart.
        """
        st = self.manifest.stat()
        stamp = (st.st_mtime_ns, st.st_size)
        if stamp != self._stamp:
            self._load()
            self._stamp = stamp
        return self._frames

    def position(self, filename: str) -> int | None:
        self.frames()
        pos = self._positions.get(filename)
        if pos is None:
            # Recording filenames carry a `+` for the UTC offset, and a query
            # string decodes `+` as a space. The UI encodes it; a hand-written
            # URL will not.
            pos = self._positions.get(filename.replace(" ", "+"))
        return pos

    def _load(self) -> None:
        records = read_manifest(self.manifest)
        # Sorted, not taken in file order: at least one capture is genuinely
        # out of order on disk. ISO timestamps with a fixed offset sort as text.
        records.sort(key=lambda r: (r.get("timestamp") or "", r["filename"]))
        stems = (
            {p.stem for p in self.audio_dir.glob("*.wav")}
            if self.audio_dir.is_dir()
            else set()
        )
        self._frames = [
            {
                "i": i,
                "filename": r["filename"],
                "t": r.get("video_offset"),
                "timestamp": r.get("timestamp"),
                "has_audio": Path(r["filename"]).stem in stems,
            }
            for i, r in enumerate(records)
        ]
        self._positions = {f["filename"]: f["i"] for f in self._frames}


def read_manifest(path: Path) -> list[dict]:
    """The manifest's records. A half-written last line (live recording) is skipped."""
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("filename"):
                out.append(rec)
    return out


def scan_library(root: Path) -> dict[str, Broadcast]:
    """Every `<root>/<host>/<dir>/classifications.jsonl`, keyed `host/dir`."""
    out: dict[str, Broadcast] = {}
    for manifest in sorted(root.glob(f"*/*/{MANIFEST}")):
        d = manifest.parent
        bid = f"{d.parent.name}/{d.name}"
        first = next(iter(read_manifest(manifest)), {})
        out[bid] = Broadcast(
            id=bid,
            root=d,
            title=first.get("video_title") or first.get("page_title") or "",
            network=first.get("network_name") or "",
        )
    return out


# ── Selection ─────────────────────────────────────────────────────────────────


def annotated_rows(bc: Broadcast) -> list[tuple[dict, dict | None]]:
    """Every frame paired with its record, with the boundary flag computed.

    A boundary is a frame whose verdict differs from either neighbour's in
    broadcast order — the segment edges, which is where placement gets checked.
    """
    frames = bc.frames()
    store = bc.store.all()
    recs = [store.get(f["filename"]) for f in frames]
    verdicts = [(r or {}).get("verdict") for r in recs]
    out = []
    for i, (f, r) in enumerate(zip(frames, recs, strict=True)):
        row = dict(f)
        prev_v = verdicts[i - 1] if i > 0 else verdicts[i]
        next_v = verdicts[i + 1] if i + 1 < len(verdicts) else verdicts[i]
        row["boundary"] = verdicts[i] != prev_v or verdicts[i] != next_v
        out.append((row, r))
    return out


FILTERS: dict[str, Callable[[dict, dict | None], bool]] = {
    "all": lambda r, v: True,
    "unlabeled": lambda r, v: not (v and (v.get("verdict") or v.get("exclude"))),
    "labeled": lambda r, v: bool(v and v.get("verdict")),
    "ad": lambda r, v: bool(v) and v.get("verdict") == "ad",
    "content": lambda r, v: bool(v) and v.get("verdict") == "content",
    "exclude": lambda r, v: bool(v and v.get("exclude")),
    "boundary": lambda r, v: r["boundary"],
    "noted": lambda r, v: bool(v and v.get("note")),
    "unfaceted": lambda r, v: (
        bool(v and v.get("verdict")) and not (v.get("video") and v.get("audio"))
    ),
    "fraught": lambda r, v: bool(v) and v.get("risk") == "fraught",
    "ad_read": lambda r, v: bool(v) and v.get("audio") == "ad_read",
}


def counts_for(rows: list[tuple[dict, dict | None]]) -> dict[str, int]:
    c = dict.fromkeys(FILTERS, 0)
    for r, v in rows:
        for k, fn in FILTERS.items():
            if fn(r, v):
                c[k] += 1
    return c


def select(bc: Broadcast, filt: str) -> tuple[list[tuple[dict, dict | None]], dict]:
    pred = FILTERS.get(filt)
    if pred is None:
        raise HTTPException(400, f"unknown filter {filt!r}")
    rows = annotated_rows(bc)
    return [(r, v) for r, v in rows if pred(r, v)], counts_for(rows)


def fill_run(bc: Broadcast, filename: str, verdict: str) -> dict:
    """Rule everything from the previous ruled frame through this one.

    Walks back in broadcast order to the last frame carrying a verdict and sets
    `verdict` on every frame after it up to and including this one. Excluded
    frames inside the run keep their exclusion and get no verdict. Returns what
    changed and what the touched frames held before, so the client can undo.
    """
    pos = bc.position(filename)
    if pos is None:
        raise HTTPException(404, f"no such frame {filename}")
    frames = bc.frames()

    def go(store: dict[str, dict]) -> dict:
        start = pos
        while start > 0 and not (store.get(frames[start - 1]["filename"]) or {}).get(
            "verdict"
        ):
            start -= 1
        if pos - start + 1 > FILL_MAX:
            raise HTTPException(
                400,
                f"run of {pos - start + 1} frames is longer than {FILL_MAX}; "
                "rule it with a range edit instead",
            )
        touched, previous = [], {}
        stamp = now()
        for f in frames[start : pos + 1]:
            fn = f["filename"]
            rec = dict(store.get(fn) or {})
            if rec.get("exclude"):
                continue
            previous[fn] = rec.get("verdict")
            rec["verdict"] = verdict
            rec["at"] = stamp
            store[fn] = rec
            touched.append(fn)
        return {
            "changed": len(touched),
            "filenames": touched,
            "first_i": start,
            "last_i": pos,
            "previous": previous,
        }

    return bc.store.mutate(go)


# ── Import ────────────────────────────────────────────────────────────────────

IMPORTED_FACETS = ("video", "audio", "risk", "care_away", "care_back")


def convert_legacy(
    verdict_rec: dict | None, facet_rec: dict | None
) -> tuple[dict, str]:
    """One old ruling plus its facets as a new record, and which case it was.

    `other` was retired as a verdict: it carried both "what the frame is" and
    "how much a miss matters", and the second already lives in the facets. Such
    a frame comes over with no verdict and its note, so it surfaces under
    `unlabeled` for a real ruling. A capture artifact is exactly what `exclude`
    exists for.
    """
    rec: dict[str, Any] = {}
    kind = "no_verdict"
    v = (verdict_rec or {}).get("verdict")
    if v in VERDICTS:
        rec["verdict"] = v
        kind = v
    elif v:
        kind = "other"
    note = (verdict_rec or {}).get("note")
    if note:
        rec["note"] = note
    for axis in IMPORTED_FACETS:
        val = (facet_rec or {}).get(axis)
        if val is not None and val != "":
            rec[axis] = val
    if (facet_rec or {}).get("artifact"):
        rec["exclude"] = True
    return rec, kind


def import_verdicts(
    bc: Broadcast,
    verdicts_path: Path,
    facets_path: Path | None,
    dataset: str,
    overwrite: bool,
    apply: bool,
) -> Counter:
    with verdicts_path.open() as fh:
        store = json.load(fh)
    if dataset not in store:
        sys.exit(f"{verdicts_path} has no dataset {dataset!r}; has {sorted(store)}")
    old_verdicts: dict[str, dict] = store[dataset]
    old_facets: dict[str, dict] = {}
    if facets_path and facets_path.exists():
        with facets_path.open() as fh:
            old_facets = json.load(fh).get("facets", {})

    stats: Counter = Counter()
    incoming: dict[str, dict] = {}
    for fn in sorted(set(old_verdicts) | set(old_facets)):
        if bc.position(fn) is None:
            stats["missing from manifest"] += 1
            continue
        rec, kind = convert_legacy(old_verdicts.get(fn), old_facets.get(fn))
        stats[f"verdict {kind}"] += 1
        if rec.get("exclude"):
            stats["artifact -> exclude"] += 1
        if rec.get("note"):
            stats["with note"] += 1
        if is_empty(rec):
            stats["empty, skipped"] += 1
            continue
        incoming[fn] = rec

    existing = bc.store.all()
    stamp = now()
    to_write = {}
    for fn, rec in incoming.items():
        if fn in existing and not overwrite:
            stats["kept existing"] += 1
            continue
        rec["at"] = stamp
        to_write[fn] = rec
    stats["written"] = len(to_write)

    if apply and to_write:
        bc.store.mutate(lambda frames: frames.update(to_write))
    return stats


# ── App ───────────────────────────────────────────────────────────────────────


class Annotate(BaseModel):
    """A patch to one frame. Absent fields are left alone; `null` clears."""

    b: str
    filename: str
    patch: dict[str, Any] = {}


class Bulk(BaseModel):
    """One patch applied to many frames in a single write.

    One write rather than one per frame: a run can be hundreds long, and
    rewriting the file each time would be both slow and a wider window for a
    concurrent editor to lose a change in.
    """

    b: str
    filenames: list[str]
    patch: dict[str, Any] = {}
    note_mode: str = "replace"


class Fill(BaseModel):
    b: str
    filename: str
    verdict: str


def safe_name(filename: str) -> str:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "bad filename")
    return filename


def app_factory(root: Path, default_broadcast: str | None = None) -> FastAPI:
    app = FastAPI(title="Broadcast annotation")
    library: dict[str, Broadcast] = scan_library(root)
    app.state.library = library

    def broadcast(bid: str) -> Broadcast:
        bc = library.get(bid)
        if bc is None:
            raise HTTPException(404, f"unknown broadcast {bid!r}")
        return bc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        default = default_broadcast or next(iter(library), "")
        return _HTML.replace("__DEFAULT_BROADCAST__", json.dumps(default)).replace(
            "__SCHEMA__", json.dumps(SCHEMA)
        )

    @app.get("/api/broadcasts")
    def broadcasts() -> JSONResponse:
        out = []
        for bid, bc in library.items():
            frames = bc.frames()
            recs = bc.store.all()
            out.append(
                {
                    "id": bid,
                    "host": bc.root.parent.name,
                    "dir": bc.root.name,
                    "title": bc.title,
                    "network": bc.network,
                    "frames": len(frames),
                    "labeled": sum(1 for r in recs.values() if r.get("verdict")),
                    "first": frames[0]["timestamp"] if frames else None,
                    "last": frames[-1]["timestamp"] if frames else None,
                }
            )
        return JSONResponse(out)

    @app.get("/api/frames")
    def frames(
        b: str, filter: str = "all", page: int = 1, per_page: int = 60
    ) -> JSONResponse:
        bc = broadcast(b)
        sel, counts = select(bc, filter)
        total = len(sel)
        per_page = max(1, min(per_page, 500))
        pages = max(1, (total + per_page - 1) // per_page)
        page = max(1, min(page, pages))
        start = (page - 1) * per_page
        rows = [dict(r, rec=v) for r, v in sel[start : start + per_page]]
        return JSONResponse(
            {
                "rows": rows,
                "total": total,
                "page": page,
                "per_page": per_page,
                "pages": pages,
                "counts": counts,
            }
        )

    @app.get("/api/sheet")
    def sheet(b: str, filter: str = "all") -> JSONResponse:
        """Every frame matching the filter, stripped to what a thumbnail paints.

        The whole selection rather than a page: its position in this list is
        what tells the client which cards page a frame lands on.
        """
        bc = broadcast(b)
        sel, counts = select(bc, filter)
        rows = [
            {"filename": r["filename"], "i": r["i"], **summarize(v)} for r, v in sel
        ]
        return JSONResponse({"rows": rows, "total": len(rows), "counts": counts})

    @app.get("/api/context")
    def context(b: str, filename: str, radius: int = 12) -> JSONResponse:
        """The frames either side of this one, in broadcast order.

        Taken from the whole broadcast rather than the current page: the point
        is to see what the broadcast was doing around a frame, and a filtered
        page is by construction a set of scattered frames.
        """
        bc = broadcast(b)
        pos = bc.position(filename)
        if pos is None:
            raise HTTPException(404, f"no such frame {filename}")
        frames = bc.frames()
        radius = max(1, min(radius, 40))
        lo, hi = max(0, pos - radius), min(len(frames), pos + radius + 1)
        store = bc.store.all()
        out = [
            {**f, **summarize(store.get(f["filename"])), "current": f["i"] == pos}
            for f in frames[lo:hi]
        ]
        return JSONResponse({"frames": out, "pos": pos - lo, "total": len(frames)})

    @app.post("/api/annotate")
    def annotate(a: Annotate) -> JSONResponse:
        bc = broadcast(a.b)
        if bc.position(a.filename) is None:
            raise HTTPException(404, f"no such frame {a.filename}")
        patch = validate_patch(a.patch)
        rec = apply_patch(bc.store.get(a.filename) or {}, patch)
        stored = bc.store.put(a.filename, rec)
        return JSONResponse({"ok": True, "filename": a.filename, "rec": stored})

    @app.post("/api/annotate/bulk")
    def annotate_bulk(a: Bulk) -> JSONResponse:
        bc = broadcast(a.b)
        if a.note_mode not in NOTE_MODES:
            raise HTTPException(400, f"note_mode must be one of {NOTE_MODES}")
        unknown = [f for f in a.filenames if bc.position(f) is None]
        if unknown:
            raise HTTPException(
                404, f"{len(unknown)} unknown frame(s), e.g. {unknown[0]}"
            )
        patch = validate_patch(a.patch)
        if not patch:
            return JSONResponse({"ok": True, "changed": 0, "cleared": 0})

        def go(frames: dict[str, dict]) -> dict:
            changed = cleared = 0
            stamp = now()
            for fn in a.filenames:
                rec = apply_patch(frames.get(fn) or {}, patch, a.note_mode)
                if is_empty(rec):
                    if frames.pop(fn, None) is not None:
                        cleared += 1
                    continue
                rec["at"] = stamp
                frames[fn] = rec
                changed += 1
            return {"ok": True, "changed": changed, "cleared": cleared}

        return JSONResponse(bc.store.mutate(go))

    @app.post("/api/fill")
    def fill(f: Fill) -> JSONResponse:
        bc = broadcast(f.b)
        if f.verdict not in VERDICTS:
            raise HTTPException(400, f"verdict must be one of {VERDICTS}")
        return JSONResponse(fill_run(bc, f.filename, f.verdict))

    @app.post("/api/reload")
    def reload() -> JSONResponse:
        """Rescan the root for new recordings. Frames and annotations already
        follow their files' mtimes, so this only matters for new directories."""
        fresh = scan_library(root)
        for bid, bc in fresh.items():
            if bid in library:
                fresh[bid] = library[bid]
        library.clear()
        library.update(fresh)
        return JSONResponse({"broadcasts": len(library)})

    @app.get("/image/{host}/{dir}/{filename}")
    def image(host: str, dir: str, filename: str) -> FileResponse:
        bc = broadcast(f"{host}/{dir}")
        path = bc.images_dir / safe_name(filename)
        if not path.is_file():
            raise HTTPException(404, f"no such frame {filename}")
        return FileResponse(path)

    @app.get("/thumb/{host}/{dir}/{filename}")
    def thumb(host: str, dir: str, filename: str) -> FileResponse:
        bc = broadcast(f"{host}/{dir}")
        path = thumbnail_for(bc, safe_name(filename))
        if path is None:
            raise HTTPException(404, f"no such frame {filename}")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/audio/{host}/{dir}/{filename}")
    def audio(host: str, dir: str, filename: str) -> FileResponse:
        bc = broadcast(f"{host}/{dir}")
        path = bc.audio_dir / (Path(safe_name(filename)).stem + ".wav")
        if not path.is_file():
            raise HTTPException(404, f"no clip for {filename}")
        return FileResponse(path, media_type="audio/wav")

    return app


def thumbnail_for(bc: Broadcast, filename: str) -> Path | None:
    """The cached thumbnail for a frame, generating it on first request.

    Written via a temp file and renamed so a concurrent request for the same
    frame never serves a half-written image.
    """
    out = bc.thumbs_dir / (Path(filename).stem + ".jpg")
    if out.is_file():
        return out
    src = bc.images_dir / filename
    if not src.is_file():
        return None
    bc.thumbs_dir.mkdir(exist_ok=True)
    tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((THUMB_WIDTH, THUMB_WIDTH))
        img.save(tmp, "JPEG", quality=80)
    tmp.replace(out)
    return out


# ── HTML ──────────────────────────────────────────────────────────────────────
# Placeholders are substituted at runtime so the JS braces need no escaping.

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Broadcast annotation</title>
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
  #controls2 { margin-bottom: 0.6rem; display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; }
  button, select {
    padding: 0.28rem 0.7rem; border: 2px solid #444; border-radius: 4px;
    background: transparent; color: #ccc; cursor: pointer; font-size: 0.8rem;
  }
  button:hover { background: #2e2e2e; }
  button.active { background: #383838; border-color: #777; color: #fff; }
  select { background: #1a1a1a; max-width: 28rem; }
  #counter { margin-left: auto; opacity: 0.5; font-size: 0.8rem; }
  .hint { font-size: 0.72rem; opacity: 0.4; margin-bottom: 0.6rem; }
  #notice { font-size: .8rem; color: #fd6; min-height: 1.1rem; }

  .grid { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .card {
    background: #242424; border-radius: 8px; overflow: hidden; position: relative;
    width: 320px; display: flex; flex-direction: column; border: 2px solid #3a3a3a;
  }
  /* Selection is a shadow, never a border colour, so the label colour survives.
     Raised so the shadow draws over the neighbouring card rather than under it. */
  .card.sel { box-shadow: 0 0 0 3px #7ab; z-index: 2; }

  /* A view switch lands on a frame somewhere in a full screen of them, so the
     new focus box blinks a few times to say where it went. Cancelled the moment
     the reviewer does anything, rather than pulsing under their hands. */
  @keyframes focusblink {
    0%, 100% { box-shadow: 0 0 0 3px #7ab; }
    50%      { box-shadow: 0 0 0 9px rgba(122, 187, 221, .9); }
  }
  .card.sel.blink, .th.focus.blink { animation: focusblink .42s ease-in-out 4; }
  /* Border carries the verdict, on the same red/green as the contact sheet. */
  .card.lab-ad      { border-color: #a02; }
  .card.lab-content { border-color: #060; }
  .card.lab-exclude { border-color: #666; border-style: dashed; }
  .card img { width: 100%; display: block; cursor: zoom-in; background: #111; min-height: 80px; }
  .fn { padding: 0.25rem 0.5rem; font-size: 0.6rem; opacity: 0.35; word-break: break-all;
        cursor: text; user-select: text; }
  .rows { padding: 0.35rem 0.5rem; display: flex; flex-direction: column; gap: 0.22rem; font-size: 0.75rem; }
  .row { display: flex; gap: 0.4rem; align-items: baseline; }
  .k { opacity: 0.42; width: 4.2rem; flex-shrink: 0; font-size: 0.7rem; }
  .badge { display: inline-block; padding: 0.08rem 0.4rem; border-radius: 3px; font-size: 0.7rem; font-weight: bold; }
  .badge.ad { background: #a02; color: #fff; }
  .badge.content { background: #060; color: #dfd; }
  .badge.exclude { background: #555; color: #ddd; }
  .badge.none { background: #333; color: #888; font-weight: normal; }
  .warn { color: #f90; }
  audio.clip { width: calc(100% - 1rem); margin: 0.15rem 0.5rem; height: 1.9rem; }

  /* Facets. Muted by default - they describe the frame, the verdict decides. */
  .facets { margin: .2rem .5rem .35rem; padding: .35rem .45rem; background: #1b1d20;
            border-radius: 4px; border-left: 2px solid #46c; font-size: .7rem; }
  .facets .fr { display: flex; gap: .3rem; align-items: center; margin-bottom: .18rem; }
  .facets .fk { opacity: .4; width: 3.6rem; flex-shrink: 0; font-size: .66rem; }
  .facets select { font-size: .68rem; padding: .05rem .2rem; border-width: 1px;
                   background: #15171a; max-width: 8.5rem; }
  .facets .care { display: flex; gap: .12rem; }
  .facets .care button { padding: 0 .3rem; font-size: .64rem; border-width: 1px; }
  .facets .care button.on { background: #46c; border-color: #46c; color: #fff; }
  .tag { font-size: .6rem; padding: 0 .25rem; border-radius: 2px; }
  .tag.fraught { background: #3a1800; color: #f90; }
  .acts { display: flex; gap: 0.25rem; padding: 0.35rem 0.5rem 0.5rem; }
  .acts button { flex: 1; padding: 0.25rem 0; font-size: 0.72rem; }
  .acts button.on { background: #383838; border-color: #999; color: #fff; }
  .note { width: 100%; background: #1a1a1a; color: #ccc; border: 1px solid #444;
          border-radius: 4px; font-size: 0.7rem; padding: 0.2rem 0.35rem; }

  #lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.95);
              flex-direction: column; align-items: center; justify-content: center;
              z-index: 100; padding: 1rem; gap: .5rem; }
  #lightbox.open { display: flex; }
  #lb-img { max-width: 92vw; max-height: 66vh; border-radius: 6px; }
  #lb-close { position: fixed; top: .5rem; right: 1rem; font-size: 2.4rem; cursor: pointer; color: #fff; }
  #lb-meta { font-size: .8rem; color: #ccc; display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
  #lb-audio { height: 2rem; }

  /* Filmstrip: the frames either side, in broadcast order. */
  #strip { display: flex; gap: 3px; overflow-x: auto; max-width: 96vw; padding: .3rem 0 .5rem; }
  .sf { flex: 0 0 auto; width: 104px; cursor: pointer; border: 2px solid transparent;
        border-radius: 4px; overflow: hidden; background: #111; position: relative; }
  .sf img { width: 100%; display: block; }
  .sf .cap { font-size: .58rem; text-align: center; padding: 1px 0; color: #ccc; }
  .sf.ad      { border-color: #a02; }
  .sf.content { border-color: #060; }
  .sf.exclude { border-color: #666; border-style: dashed; }
  .sf.cur     { border-color: #7ab; box-shadow: 0 0 0 2px #7ab; }

  /* Contact sheet: the whole selection at a glance, for picking out segment
     edges. Border carries the verdict, the corner dots carry note / fraught. */
  #sheet { display: flex; flex-wrap: wrap; gap: 2px; }
  .th { position: relative; cursor: pointer; border: 2px solid #333;
        border-radius: 3px; overflow: hidden; background: #111; line-height: 0; }
  .th img { display: block; width: 100%; height: 100%; object-fit: cover; }
  .th.ad      { border-color: #a02; }
  .th.content { border-color: #060; }
  .th.exclude { border-color: #666; border-style: dashed; }
  /* Same treatment as a selected card: a shadow over the label colour, raised
     so the packed grid does not clip it. */
  .th.focus { box-shadow: 0 0 0 3px #7ab; z-index: 2; }
  .th .dot { position: absolute; top: 2px; right: 2px; width: 7px; height: 7px;
             border-radius: 50%; border: 1px solid #000a; }
  .th .dot.note { background: #7ab; }
  .th .dot.fraught { background: #fb3; right: auto; left: 2px; }
  .th .idx { position: absolute; bottom: 0; left: 0; right: 0; font-size: .5rem;
             text-align: center; background: #000a; color: #ddd; line-height: 1.3; }
  #size { width: 8rem; }
  .legend { font-size: .68rem; opacity: .45; display: flex; gap: .9rem; flex-wrap: wrap;
            margin-bottom: .5rem; }

  /* Range selection. The anchor is one end of a range not yet closed. */
  .card.anchor, .th.anchor { box-shadow: 0 0 0 3px #fb3; z-index: 2; }
  .card.inrange, .th.inrange { outline: 2px solid #fb3; outline-offset: -2px; }
  #selbar {
    position: sticky; top: 0; z-index: 20; display: none;
    background: #2b2b1a; border: 1px solid #6a5; border-radius: 6px;
    padding: .45rem .8rem; margin-bottom: .6rem; font-size: .82rem;
    align-items: center; gap: .8rem;
  }
  #selbar.on { display: flex; }
  #selbar b { color: #fd6; }

  /* Bulk editor. Same fields as a card, but nothing is written until Save. */
  #bulk { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.9);
          z-index: 200; align-items: center; justify-content: center; padding: 1rem; }
  #bulk.open { display: flex; }
  .bulkbox { background: #202020; border: 1px solid #444; border-radius: 8px;
             padding: 1rem; max-width: 96vw; max-height: 94vh; overflow-y: auto;
             display: flex; flex-direction: column; gap: .7rem; min-width: 40rem; }
  .bulkbox h2 { margin: 0; font-size: 1rem; font-weight: normal; }
  .bulkbox h2 b { color: #7ab; }
  #bulkstrip { display: flex; flex-wrap: wrap; gap: 3px; max-height: 36vh;
               overflow-y: auto; padding: .2rem; background: #161616; border-radius: 4px; }
  .choice { display: flex; gap: .35rem; flex-wrap: wrap; align-items: center; }
  .choice button.on { background: #7ab; border-color: #7ab; color: #04121a; font-weight: bold; }
  #bulknote { width: 100%; min-height: 3.2rem; background: #1a1a1a; color: #ddd;
              border: 1px solid #444; border-radius: 4px; padding: .4rem; font: inherit;
              font-size: .8rem; }
  .bulkrow { display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
  .bulkrow .lbl { font-size: .75rem; opacity: .6; width: 5.5rem; }
  #bulksave { background: #2a5; border-color: #2a5; color: #041; font-weight: bold; }
  .muted { font-size: .72rem; opacity: .45; }
</style>
</head>
<body>
<h1>Broadcast annotation</h1>
<div class="bar" id="bar"></div>
<div id="controls">
  <select id="broadcast" title="Which recording"></select>
  <button data-f="unlabeled">Unlabeled</button>
  <button data-f="labeled">Labeled</button>
  <button data-f="ad">Ad</button>
  <button data-f="content">Content</button>
  <button data-f="exclude">Excluded</button>
  <button data-f="boundary" title="Verdict differs from a neighbour's">Boundaries</button>
  <button data-f="noted">Noted</button>
  <button data-f="unfaceted" title="Ruled, but video or audio facet missing">Unfaceted</button>
  <button data-f="fraught" title="Correct, but risky to train on">Fraught</button>
  <button data-f="ad_read">Ad reads</button>
  <button data-f="all">All</button>
  <button id="reload" title="Rescan the root for new recordings">⟳</button>
  <span id="counter"></span>
</div>
<div id="controls2">
  <button id="viewtoggle" title="Switch between cards and a scannable contact sheet"></button>
  <label id="sizewrap" style="font-size:.75rem;opacity:.6;display:none">
    thumb <input type="range" id="size" min="48" max="240" step="8" value="112">
  </label>
  <span id="notice"></span>
</div>
<div class="hint">
  <b>a</b> ad · <b>r</b> content/racing · <b>o</b> toggle exclude (undecidable: bumper, black, capture artifact) ·
  <b>x</b> clear the ruling · <b>A</b>/<b>R</b> fill from the previous ruled frame through this one · <b>u</b> undo the last fill ·
  <b>n</b> edit the note · <b>←/→</b> or <b>j/k</b> move one card, <b>↑/↓</b> move a row · <b>Enter</b> or click a frame to open it in context ·
  <b>s</b> shows it in the contact sheet · shift+click two frames then <b>e</b> to edit the run.
  In context view, <b>←/→</b> walk the broadcast, <b>a</b>/<b>r</b>/<b>o</b>/<b>x</b>/<b>A</b>/<b>R</b> rule the frame on screen, <b>space</b> plays the clip.
  <br>Border says what the frame is — <b style="color:#f55">red</b> ad, <b style="color:#4d4">green</b> content,
  <b style="color:#999">dashed grey</b> excluded. Annotations save to <code>annotations.json</code> in the recording's directory.
</div>
<div id="selbar">
  <span id="selmsg"></span>
  <button id="seledit">Edit selected…</button>
  <button id="selclear">Clear selection</button>
  <span class="muted">shift+click two frames to pick a range · <b>e</b> to edit · Esc to clear</span>
</div>
<div class="legend" id="legend" style="display:none">
  <span>border: <b style="color:#f55">red</b> ad · <b style="color:#4d4">green</b> content · <b style="color:#999">dashed</b> excluded · none unlabeled</span>
  <span>dots: <b style="color:#7ab">blue</b> has a note · <b style="color:#fb3">amber</b> fraught</span>
  <span>click a thumbnail to focus it, click the focused one (or <b>Enter</b>) to open its cards page</span>
  <span>arrows move · <b>A</b>/<b>R</b> fill to the focused frame · shift+click two frames then <b>e</b> to edit the run</span>
</div>
<div class="grid" id="grid"></div>
<div id="sheet" style="display:none"></div>
<div id="pager" style="margin-top:1rem;display:flex;gap:.5rem;align-items:center"></div>
<div id="bulk">
  <div class="bulkbox">
    <h2>Editing <b id="bulkcount"></b> frames <span class="muted" id="bulkrange"></span></h2>
    <div id="bulkstrip"></div>
    <div id="bulkfields"></div>
    <div class="bulkrow" style="align-items:flex-start">
      <span class="lbl">note</span>
      <div style="flex:1;min-width:22rem">
        <div class="choice" id="bulknotemode">
          <button data-m="">leave as is</button>
          <button data-m="replace">replace</button>
          <button data-m="fill">only where empty</button>
          <button data-m="append">append</button>
          <button data-m="__clear__">clear</button>
        </div>
        <textarea id="bulknote" placeholder="typing here switches off &quot;leave as is&quot;"></textarea>
        <div class="muted" id="bulknotehint"></div>
      </div>
    </div>
    <div class="bulkrow" style="justify-content:flex-end">
      <span class="muted" id="bulkwarn"></span>
      <button id="bulkcancel">Cancel</button>
      <button id="bulksave">Save</button>
    </div>
  </div>
</div>

<div id="lightbox">
  <span id="lb-close" title="Close (Esc)">&times;</span>
  <img id="lb-img" src="" alt="">
  <div id="lb-meta"></div>
  <audio id="lb-audio" controls preload="none"></audio>
  <div id="strip"></div>
</div>

<script>
const SCHEMA = __SCHEMA__;   // facet vocabularies
let B = __DEFAULT_BROADCAST__;
let FILTER = "all";
let PAGE = 1;
let VIEW = "cards";      // "cards" | "sheet"
let PER_PAGE = 60;
let THUMB = 112;
let FOCUS = null;        // filename to highlight after a jump from the sheet
let ROWS = [];           // the current cards page
let SHEET = [];          // the whole filtered selection, contact-sheet view
let NOTICE = "";         // one-shot message shown under the sheet
let WIDEN_FOR_FOCUS = false;  // this load was asked to reveal a specific frame
let SEL = 0;
let LAST_FILL = null;    // {filenames, previous} of the last fill, for `u`

// ── URL state ───────────────────────────────────────────────────────────────
// Every option lives in the query string, so any view is linkable and survives
// a reload.

function readUrl() {
  const q = new URLSearchParams(location.search);
  B = q.get("b") || B;
  FILTER = q.get("filter") || FILTER;
  PAGE = Math.max(1, parseInt(q.get("page") || "1", 10) || 1);
  VIEW = q.get("view") === "sheet" ? "sheet" : "cards";
  PER_PAGE = Math.min(500, Math.max(1, parseInt(q.get("per_page") || "60", 10) || 60));
  THUMB = Math.min(240, Math.max(48, parseInt(q.get("thumb") || "112", 10) || 112));
  FOCUS = q.get("frame");
}

function syncUrl(push) {
  const q = new URLSearchParams();
  q.set("b", B);
  q.set("filter", FILTER);
  q.set("view", VIEW);
  if (VIEW === "cards") { q.set("page", PAGE); q.set("per_page", PER_PAGE); }
  else q.set("thumb", THUMB);
  if (FOCUS) q.set("frame", FOCUS);
  const url = location.pathname + "?" + q.toString();
  if (push) history.pushState(null, "", url); else history.replaceState(null, "", url);
}

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const num = (x, d=1) => (x === null || x === undefined) ? "–" : (typeof x === "number" ? x.toFixed(d) : x);
const thumbUrl = (fn) => `/thumb/${B}/${encodeURIComponent(fn)}`;
const imageUrl = (fn) => `/image/${B}/${encodeURIComponent(fn)}`;
const audioUrl = (fn) => `/audio/${B}/${encodeURIComponent(fn)}`;
const bq = () => `b=${encodeURIComponent(B)}`;

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                              body: JSON.stringify(body) });
  if (!r.ok) { notice(await r.text()); return null; }
  return r.json();
}

// Whatever a frame's ruling is, as a CSS class shared by cards, tiles and strip.
function labClass(rec) {
  if (!rec) return "";
  if (rec.verdict) return rec.verdict;
  if (rec.exclude) return "exclude";
  return "";
}

let NOTICE_TIMER = null;
function notice(text) {
  $("#notice").textContent = text || "";
  if (NOTICE_TIMER) clearTimeout(NOTICE_TIMER);
  if (text) NOTICE_TIMER = setTimeout(() => { $("#notice").textContent = ""; }, 8000);
}

// ── Broadcasts ──────────────────────────────────────────────────────────────

async function loadBroadcasts() {
  const r = await fetch("/api/broadcasts");
  const list = await r.json();
  const byHost = {};
  for (const b of list) (byHost[b.host] ??= []).push(b);
  $("#broadcast").innerHTML = Object.entries(byHost).map(([host, bs]) =>
    `<optgroup label="${esc(host)}">` + bs.map(b =>
      `<option value="${esc(b.id)}">${esc(b.dir)} (${b.frames}, ${b.labeled} ruled)</option>`
    ).join("") + `</optgroup>`).join("");
  $("#broadcast").value = B;
}

// ── Loading ─────────────────────────────────────────────────────────────────

async function load() {
  // Range indices point into the list being shown, so any navigation - a new
  // page, filter, broadcast or view - makes them meaningless.
  clearRange();
  syncUrl(false);
  $("#viewtoggle").textContent = VIEW === "sheet" ? "▤ Cards view" : "▦ Contact sheet";
  $("#sizewrap").style.display = VIEW === "sheet" ? "" : "none";
  $("#legend").style.display = VIEW === "sheet" ? "" : "none";
  $("#grid").style.display = VIEW === "sheet" ? "none" : "";
  $("#sheet").style.display = VIEW === "sheet" ? "" : "none";
  if (VIEW === "sheet") return loadSheet();

  const r = await fetch(`/api/frames?${bq()}&filter=${FILTER}&page=${PAGE}&per_page=${PER_PAGE}`);
  if (!r.ok) { $("#grid").innerHTML = `<p class="warn">${esc(await r.text())}</p>`; return; }
  const d = await r.json();
  ROWS = d.rows;
  PAGE = d.page;   // clamped by the server
  // A jump from the sheet names a frame; land on it rather than the top.
  const at = FOCUS ? ROWS.findIndex(x => x.filename === FOCUS) : -1;
  SEL = at >= 0 ? at : 0;
  // Arriving with no frame named, the top card is where the view is, so the
  // sheet has somewhere to scroll to if the toggle is hit straight away.
  if (ROWS[SEL]) { FOCUS = ROWS[SEL].filename; syncUrl(false); }
  render(d);
  if (at >= 0) document.querySelectorAll(".card")[at]
    ?.scrollIntoView({ block: "center", behavior: "smooth" });
  if (BLINK_NEXT) { BLINK_NEXT = false; blinkFocus(".card.sel"); }
}

async function loadSheet() {
  const r = await fetch(`/api/sheet?${bq()}&filter=${FILTER}`);
  if (!r.ok) { $("#sheet").innerHTML = `<p class="warn">${esc(await r.text())}</p>`; return; }
  const d = await r.json();
  // Arriving from the filmstrip, the frame may sit outside the current filter -
  // the strip walks the whole broadcast. Widen to `all` once so the jump lands
  // on the frame instead of silently on nothing. Only for that deliberate
  // "show me this frame" move: a reload after a sweep must not yank the filter
  // out from under a run that has just stopped matching it.
  const widen = WIDEN_FOR_FOCUS;
  WIDEN_FOR_FOCUS = false;
  if (widen && FOCUS && FILTER !== "all"
      && !d.rows.some(n => n.filename === FOCUS)) {
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
    <div class="th ${labClass(n)} ${n.filename === FOCUS ? "focus" : ""}"
         style="width:${w}px;height:${h}px"
         onclick="event.shiftKey ? shiftPick(${k}) : sheetClick(${k})"
         title="i=${n.i}${n.verdict ? " · " + esc(n.verdict) : ""}${n.exclude ? " · excluded" : ""}">
      <img src="${thumbUrl(n.filename)}" loading="lazy">
      ${n.has_note ? `<span class="dot note"></span>` : ""}
      ${n.fraught ? `<span class="dot fraught"></span>` : ""}
      <span class="idx">${n.i}</span>
    </div>`).join("");
  if (FOCUS) document.querySelector(".th.focus")
    ?.scrollIntoView({ block: "center", behavior: "smooth" });
  if (BLINK_NEXT) { BLINK_NEXT = false; blinkFocus(".th.focus"); }
}

// A plain click moves the focus; clicking the frame already focused is what
// opens it. Two meanings on one button, ordered so the destructive one - losing
// your place in a sheet of thousands - needs the deliberate second click.
function sheetClick(k) {
  const n = SHEET[k];
  if (!n) return;
  if (n.filename === FOCUS) return jumpTo(k);
  stopBlink();
  FOCUS = n.filename;
  syncUrl(false);
  paintSheetFocus();
}

function paintSheetFocus() {
  document.querySelectorAll(".th").forEach((el, i) =>
    el.classList.toggle("focus", !!SHEET[i] && SHEET[i].filename === FOCUS));
}

// Position within the filtered selection decides which cards page holds it.
function jumpTo(k) {
  stopBlink();
  const n = SHEET[k];
  if (!n) return;
  FOCUS = n.filename;
  PAGE = Math.floor(k / PER_PAGE) + 1;
  VIEW = "cards";
  BLINK_NEXT = true;
  syncUrl(true);
  load();
}

function stats(c) {
  $("#bar").innerHTML = [
    ["unlabeled", "unlabeled", "bad"], ["labeled", "ruled", "good"],
    ["ad", "ad", ""], ["content", "content", ""], ["exclude", "excluded", ""],
    ["boundary", "boundaries", ""], ["noted", "noted", ""],
    ["unfaceted", "unfaceted", ""], ["fraught", "fraught", ""],
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

// ── Cards ───────────────────────────────────────────────────────────────────

function badge(v, cls) {
  return `<span class="badge ${cls || v || "none"}">${esc(v ?? "—")}</span>`;
}

function facetSelect(idx, axis, value, vocab) {
  const opts = ['<option value="">—</option>'].concat(
    Object.entries(vocab || {}).map(([k, help]) =>
      `<option value="${k}" title="${esc(help)}"${k === value ? " selected" : ""}>${esc(k)}</option>`));
  return `<select onchange="setField(${idx}, {${axis}: this.value || null})"
            onclick="event.stopPropagation()">${opts.join("")}</select>`;
}

function careRow(idx, key, value) {
  return `<div class="care">` + [0, 1, 2, 3].map(n =>
    `<button class="${n === value ? "on" : ""}" title="click again to clear"
       onclick="event.stopPropagation();setField(${idx}, {${key}: ${n === value ? "null" : n}})">${n}</button>`
  ).join("") + `</div>`;
}

function facetBlock(r, idx) {
  const f = r.rec || {};
  return `<div class="facets" onclick="event.stopPropagation()">
    <div class="fr"><span class="fk">video</span>${facetSelect(idx, "video", f.video, SCHEMA.video)}</div>
    <div class="fr"><span class="fk">audio</span>${facetSelect(idx, "audio", f.audio, SCHEMA.audio)}</div>
    <div class="fr"><span class="fk">care away</span>${careRow(idx, "care_away", f.care_away)}
      <span class="fk" style="width:auto;margin-left:.3rem">back</span>${careRow(idx, "care_back", f.care_back)}</div>
    <div class="fr"><span class="fk">risk</span>${facetSelect(idx, "risk", f.risk, SCHEMA.risk)}
      ${f.risk === "fraught" ? '<span class="tag fraught">fraught</span>' : ""}</div>
  </div>`;
}

function card(r, idx) {
  const rec = r.rec || {};
  const rows = [];
  rows.push(`<div class="row"><span class="k">frame</span><span style="opacity:.6">i=${r.i} · t=${num(r.t)}s · ${esc((r.timestamp || "").replace("T", " ").slice(0, 19))}</span></div>`);
  rows.push(`<div class="row"><span class="k">ruling</span>${badge(rec.verdict)}${rec.exclude ? " " + badge("excluded", "exclude") : ""}${r.boundary ? ' <span class="warn" style="font-size:.66rem">edge</span>' : ""}</div>`);

  return `<div class="card ${labClass(rec) ? "lab-" + labClass(rec) : ""}" data-idx="${idx}"
       onclick="event.shiftKey ? shiftPick(${idx}) : focusCard(${idx})">
    <img src="${thumbUrl(r.filename)}" loading="lazy"
         title="Click to see this frame in context"
         onclick="event.stopPropagation();zoom('${r.filename}')">
    <!-- Selecting the filename to copy it must not move the card selection. -->
    <div class="fn" onclick="event.stopPropagation()"
         onmousedown="event.stopPropagation()">${esc(r.filename)}</div>
    <div class="rows">${rows.join("")}</div>
    ${facetBlock(r, idx)}
    ${r.has_audio ? `<audio class="clip" controls preload="none"
         onclick="event.stopPropagation()" src="${audioUrl(r.filename)}"></audio>` : ""}
    <!-- Acting on a card moves the selection to it, so the keyboard carries on
         from where the mouse left off rather than from the last card walked. -->
    <div class="acts">
      <button class="${rec.verdict === "ad" ? "on" : ""}" onclick="event.stopPropagation();focusCard(${idx});setField(${idx},{verdict:'ad'})">ad</button>
      <button class="${rec.verdict === "content" ? "on" : ""}" onclick="event.stopPropagation();focusCard(${idx});setField(${idx},{verdict:'content'})">content</button>
      <button class="${rec.exclude ? "on" : ""}" onclick="event.stopPropagation();focusCard(${idx});setField(${idx},{exclude:${rec.exclude ? "null" : "true"}})">exclude</button>
      <button onclick="event.stopPropagation();focusCard(${idx});setField(${idx},{verdict:null,exclude:null})" title="clear the ruling">×</button>
    </div>
    <div style="padding:0 .5rem .5rem"><input class="note" placeholder="note"
      value="${esc(rec.note ?? "")}" onclick="event.stopPropagation()"
      onfocus="focusCard(${idx})" onchange="setField(${idx}, {note: this.value})"
      onkeydown="if (event.key === 'Enter' || event.key === 'Escape') this.blur()"></div>
  </div>`;
}

function paint() {
  document.querySelectorAll(".card").forEach((el, i) => el.classList.toggle("sel", i === SEL));
  paintRange();   // a single-card re-render drops its range classes
  const el = document.querySelectorAll(".card")[SEL];
  if (el) el.scrollIntoView({ block: "nearest" });
}

// The selected card is the frame the view is "on", so it is what the contact
// sheet scrolls to and what the URL names. Keeping FOCUS in step here is what
// makes the view toggle carry the current card rather than the last jump.
function setSel(i) {
  stopBlink();   // the reviewer is moving the focus themselves now
  SEL = Math.max(0, Math.min(i, ROWS.length - 1));
  const r = ROWS[SEL];
  if (r) { FOCUS = r.filename; syncUrl(false); }
  paint();
}

// Named focusCard, not select: an inline handler puts the element on the
// scope chain, and <input> already has a select() that grabs its own text -
// which would silently shadow the global and make the note field do nothing.
function focusCard(i) { setSel(i); }

function rerenderCard(idx) {
  const el = document.querySelectorAll(".card")[idx];
  if (el) { el.outerHTML = card(ROWS[idx], idx); paint(); }
}

// ── Writes ──────────────────────────────────────────────────────────────────
// A click on a card is the write. Whatever is on screen for that frame - its
// card, its tile, its filmstrip entry - is refreshed from the reply.

async function annotate(filename, patch) {
  const d = await post("/api/annotate", { b: B, filename, patch });
  if (!d) return null;
  const rec = d.rec;
  const idx = ROWS.findIndex(r => r.filename === filename);
  if (idx >= 0) { ROWS[idx].rec = rec; rerenderCard(idx); }
  const s = SHEET.findIndex(n => n.filename === filename);
  if (s >= 0) {
    Object.assign(SHEET[s], {
      verdict: rec?.verdict ?? null, exclude: !!rec?.exclude,
      has_note: !!rec?.note, fraught: rec?.risk === "fraught",
    });
    if (VIEW === "sheet") loadSheet();
  }
  const c = CTX.findIndex(n => n.filename === filename);
  if (c >= 0) {
    Object.assign(CTX[c], { verdict: rec?.verdict ?? null, exclude: !!rec?.exclude, has_note: !!rec?.note });
    if ($("#lightbox").classList.contains("open")) showCtx();
  }
  return rec;
}

function setField(idx, patch) {
  const r = ROWS[idx];
  if (r) annotate(r.filename, patch);
}

// Everything from the previous ruled frame through `filename`, in one write.
async function fillTo(filename, verdict) {
  if (!filename) return;
  const d = await post("/api/fill", { b: B, filename, verdict });
  if (!d) return;
  LAST_FILL = { filenames: d.filenames, previous: d.previous };
  notice(`filled ${d.changed} frame${d.changed === 1 ? "" : "s"} as ${verdict}, i=${d.first_i}…${d.last_i} — u to undo`);
  await refreshInPlace();
}

async function undoFill() {
  if (!LAST_FILL) return notice("nothing to undo");
  const { filenames, previous } = LAST_FILL;
  LAST_FILL = null;
  const fresh = filenames.filter(fn => !previous[fn]);
  if (fresh.length) await post("/api/annotate/bulk", { b: B, filenames: fresh, patch: { verdict: null } });
  for (const fn of filenames) if (previous[fn]) await post("/api/annotate", { b: B, filename: fn, patch: { verdict: previous[fn] } });
  notice(`undid the fill on ${filenames.length} frame${filenames.length === 1 ? "" : "s"}`);
  await refreshInPlace();
}

// Reload whatever is on screen without losing the place in it.
async function refreshInPlace() {
  const keep = FOCUS;
  if ($("#lightbox").classList.contains("open") && CTX[CTX_POS]) {
    const fn = CTX[CTX_POS].filename;
    const r = await fetch(`/api/context?${bq()}&filename=${encodeURIComponent(fn)}&radius=12`);
    if (r.ok) { const d = await r.json(); CTX = d.frames; CTX_POS = d.pos; showCtx(); }
  }
  FOCUS = keep;
  await load();
}

// ── Range selection ─────────────────────────────────────────────────────────
// Shift+click one frame, then another, to take everything between them. Two
// explicit clicks rather than "extend from wherever the focus is", because in
// the sheet a plain click navigates away, so there is no way to place a start
// without leaving. The same gesture then means the same thing in both views.
let ANCHOR = null;   // index of an open range's first end
let RANGE = null;    // {lo, hi, end} once closed; `end` is the second click,
                     // which is where attention was last and so where the
                     // focus returns when the editor closes.

// Whichever list the current view is showing; range indices are into this.
function viewList() { return VIEW === "sheet" ? SHEET : ROWS; }

function shiftPick(i) {
  stopBlink();
  if (ANCHOR === null || RANGE) {          // start over
    ANCHOR = i; RANGE = null;
  } else {
    RANGE = { lo: Math.min(ANCHOR, i), hi: Math.max(ANCHOR, i), end: i };
  }
  paintRange();
}

function rangeEndFrame() {
  return RANGE ? viewList()[RANGE.end] : null;
}

function clearRange() { ANCHOR = null; RANGE = null; paintRange(); }

function rangeFrames() {
  if (!RANGE) return [];
  return viewList().slice(RANGE.lo, RANGE.hi + 1);
}

function paintRange() {
  const cls = VIEW === "sheet" ? ".th" : ".card";
  document.querySelectorAll(cls).forEach((el, i) => {
    el.classList.toggle("anchor", ANCHOR !== null && !RANGE && i === ANCHOR);
    el.classList.toggle("inrange", !!RANGE && i >= RANGE.lo && i <= RANGE.hi);
  });
  const bar = $("#selbar");
  if (RANGE) {
    const n = RANGE.hi - RANGE.lo + 1;
    const list = viewList();
    $("#selmsg").innerHTML = `<b>${n}</b> frames selected ` +
      `<span class="muted">(i=${list[RANGE.lo]?.i} … ${list[RANGE.hi]?.i})</span>`;
    bar.classList.add("on");
  } else if (ANCHOR !== null) {
    $("#selmsg").innerHTML = `range starts at <b>i=${viewList()[ANCHOR]?.i}</b> — shift+click the other end`;
    bar.classList.add("on");
  } else {
    bar.classList.remove("on");
  }
}

// ── Focus blink ─────────────────────────────────────────────────────────────
// Only a view switch blinks: arriving in the other view, the frame you were on
// is one of a screenful, and a static box is easy to miss. Anything the
// reviewer does that would move the focus cancels it.
let BLINK_NEXT = false;   // the next render is the far side of a view switch
let BLINK_TIMER = null;

function blinkFocus(selector) {
  stopBlink();
  const el = document.querySelector(selector);
  if (!el) return;
  el.classList.add("blink");
  BLINK_TIMER = setTimeout(stopBlink, 1800);
}

function stopBlink() {
  if (BLINK_TIMER) { clearTimeout(BLINK_TIMER); BLINK_TIMER = null; }
  document.querySelectorAll(".blink").forEach(el => el.classList.remove("blink"));
}

// How many tiles sit on one row of a wrapped grid, measured rather than derived
// from the CSS width, so up/down follow what is actually on screen.
function columnCount(selector) {
  const tiles = document.querySelectorAll(selector);
  if (tiles.length < 2) return 1;
  const top = tiles[0].offsetTop;
  let n = 0;
  for (const t of tiles) {
    if (t.offsetTop !== top) break;
    n++;
  }
  return Math.max(1, n);
}

// The sheet's focus is a filename rather than an index, since the list it
// indexes is rebuilt on every load; the index is derived when needed.
function sheetIndex() {
  return SHEET.findIndex(n => n.filename === FOCUS);
}

function setSheetSel(k) {
  const n = SHEET[Math.max(0, Math.min(k, SHEET.length - 1))];
  if (!n) return;
  stopBlink();
  FOCUS = n.filename;
  syncUrl(false);
  paintSheetFocus();
  document.querySelector(".th.focus")?.scrollIntoView({ block: "nearest" });
}
function go(p) { PAGE = p; FOCUS = null; syncUrl(true); load(); }

// ── Lightbox: one frame in the context of its neighbours ────────────────────
let CTX = [];      // the neighbouring frames, in broadcast order
let CTX_POS = 0;   // which of them is on screen

async function zoom(filename) {
  stopBlink();
  const r = await fetch(`/api/context?${bq()}&filename=${encodeURIComponent(filename)}&radius=12`);
  if (!r.ok) return;
  const d = await r.json();
  CTX = d.frames; CTX_POS = d.pos;
  $("#lightbox").classList.add("open");
  showCtx();
}

function showCtx() {
  const f = CTX[CTX_POS];
  if (!f) return;
  $("#lb-img").src = imageUrl(f.filename);
  const bits = [`i=${f.i}`, f.t != null ? `t=${f.t.toFixed(1)}s` : "",
                f.verdict ? `ruled ${f.verdict}` : "unruled", f.exclude ? "excluded" : "",
                f.has_note ? "has note" : ""].filter(Boolean);
  $("#lb-meta").innerHTML = bits.map(esc).join(" &middot; ") +
    ` <button onclick="openInSheet()" title="Show this frame in the contact sheet (s)">▦ in sheet</button>` +
    `<span style="opacity:.45">&nbsp;←/→ step · a/r/o/x rule · A/R fill · space plays · Esc closes</span>`;

  const au = $("#lb-audio");
  if (f.has_audio) {
    au.style.display = "";
    if (!au.src.endsWith(audioUrl(f.filename))) au.src = audioUrl(f.filename);
  } else {
    au.style.display = "none";
    au.removeAttribute("src");
  }

  $("#strip").innerHTML = CTX.map((n, k) => `
    <div class="sf ${labClass(n)} ${k === CTX_POS ? "cur" : ""}" onclick="stepTo(${k})">
      <img src="${thumbUrl(n.filename)}" loading="lazy">
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
function showInSheet(filename) {
  if (!filename) return;
  FOCUS = filename;
  VIEW = "sheet";
  BLINK_NEXT = true;
  WIDEN_FOR_FOCUS = true;
  syncUrl(true);
  load();
}

function openInSheet() {
  const f = CTX[CTX_POS];
  if (!f) return;
  closeLightbox();
  showInSheet(f.filename);
}

function closeLightbox() {
  $("#lightbox").classList.remove("open");
  const au = $("#lb-audio");
  au.pause();
  au.removeAttribute("src");
}

// ── Bulk editor ─────────────────────────────────────────────────────────────
// Nothing here is written until Save, unlike a card where a click is the write.
// A sweep can touch hundreds of frames, so it gets a chance to be looked at
// first, and Cancel has to be a real way out.
const LEAVE = "__leave__", CLEAR = "__clear__";
let BULK = {};        // field -> LEAVE | CLEAR | value
let BULK_NOTE = "";   // "" leave as is · replace/fill/append/__clear__
let BULK_NOTED = 0;   // how many of the selection already carry a note

const BULK_FIELDS = [
  ["verdict", ["ad", "content"]],
  ["exclude", [true]],
  ["video", Object.keys(SCHEMA.video)],
  ["audio", Object.keys(SCHEMA.audio)],
  ["risk", Object.keys(SCHEMA.risk)],
  ["care_away", [0, 1, 2, 3]],
  ["care_back", [0, 1, 2, 3]],
];

function bulkFieldsHtml() {
  return BULK_FIELDS.map(([f, vals]) => `
    <div class="bulkrow"><span class="lbl">${esc(f.replace("_", " "))}</span>
      <div class="choice" data-f="${f}">
        <button data-v="${LEAVE}">leave as is</button>
        ${vals.map(v => `<button data-v="${esc(String(v))}">${f === "exclude" ? "exclude" : esc(String(v))}</button>`).join("")}
        <button data-v="${CLEAR}">clear</button>
      </div></div>`).join("");
}

function openBulk() {
  const frames = rangeFrames();
  if (!frames.length) return;
  stopBlink();
  BULK = Object.fromEntries(BULK_FIELDS.map(([f]) => [f, LEAVE]));
  BULK_NOTE = "";
  $("#bulknote").value = "";
  $("#bulkcount").textContent = frames.length;
  $("#bulkrange").textContent = `i=${frames[0].i} … ${frames[frames.length - 1].i}`;
  const ruled = frames.filter(f => (f.verdict ?? f.rec?.verdict)).length;
  const noted = frames.filter(f => (f.has_note ?? !!f.rec?.note)).length;
  BULK_NOTED = noted;
  $("#bulkwarn").textContent = [
    ruled ? `${ruled} already ruled` : "",
    noted ? `${noted} already have a note` : "",
  ].filter(Boolean).join(" · ");
  $("#bulkstrip").innerHTML = frames.map(f => {
    const rec = f.rec ?? f;
    return `<div class="th ${labClass(rec)}" style="width:88px;height:50px" title="i=${f.i}">
      <img src="${thumbUrl(f.filename)}" loading="lazy">
      <span class="idx">${f.i}</span></div>`;
  }).join("");
  $("#bulkfields").innerHTML = bulkFieldsHtml();
  document.querySelectorAll("#bulkfields .choice button").forEach(b =>
    b.onclick = () => { BULK[b.parentElement.dataset.f] = b.dataset.v; paintBulkChoice(); });
  paintBulkChoice();
  $("#bulk").classList.add("open");
}

function paintBulkChoice() {
  document.querySelectorAll("#bulkfields .choice").forEach(c =>
    c.querySelectorAll("button").forEach(b =>
      b.classList.toggle("on", b.dataset.v === String(BULK[c.dataset.f]))));
  document.querySelectorAll("#bulknotemode button").forEach(b =>
    b.classList.toggle("on", b.dataset.m === BULK_NOTE));
  // Spell out what the chosen mode will do to the notes already there, since
  // that is the part a sweep can quietly destroy.
  const n = BULK_NOTED, total = rangeFrames().length;
  const hint = {
    "": "notes left untouched",
    replace: n ? `overwrites the note on ${n} frame${n === 1 ? "" : "s"}`
                : "sets the note on all of them",
    fill: `writes only to the ${total - n} with no note; leaves ${n} as ${n === 1 ? "it is" : "they are"}`,
    append: n ? `adds after " || " on ${n}; sets it plain on the other ${total - n}`
              : "sets the note on all of them",
    [CLEAR]: n ? `removes the note from ${n} frame${n === 1 ? "" : "s"}` : "nothing to clear",
  }[BULK_NOTE];
  $("#bulknotehint").textContent = hint || "";
}

function closeBulk() {
  const end = rangeEndFrame();
  $("#bulk").classList.remove("open");
  if (!end) return;
  FOCUS = end.filename;
  syncUrl(false);
  if (VIEW === "sheet") {
    paintSheetFocus();
    document.querySelector(".th.focus")?.scrollIntoView({ block: "center", behavior: "smooth" });
  } else {
    SEL = RANGE.end;
    paint();
    document.querySelectorAll(".card")[SEL]
      ?.scrollIntoView({ block: "center", behavior: "smooth" });
  }
}

// The bulk choices as a patch: LEAVE is absent, CLEAR is null, a value is
// itself (with the string the button carried turned back into its type).
function bulkPatch() {
  const patch = {};
  for (const [f, vals] of BULK_FIELDS) {
    const v = BULK[f];
    if (v === LEAVE) continue;
    if (v === CLEAR) { patch[f] = null; continue; }
    patch[f] = vals.find(x => String(x) === v) ?? v;
  }
  if (BULK_NOTE === CLEAR) patch.note = null;
  else if (BULK_NOTE) patch.note = $("#bulknote").value;
  return patch;
}

async function saveBulk() {
  const frames = rangeFrames();
  const patch = bulkPatch();
  if (!frames.length || !Object.keys(patch).length) return closeBulk();
  const body = {
    b: B, filenames: frames.map(f => f.filename), patch,
    note_mode: (BULK_NOTE && BULK_NOTE !== CLEAR) ? BULK_NOTE : "replace",
  };
  const r = await fetch("/api/annotate/bulk", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { $("#bulkwarn").textContent = await r.text(); return; }
  closeBulk();
  clearRange();
  load();
}

// One clip at a time. A page of cards holds dozens of players and the context
// view another, and two commentaries over each other tell you nothing about
// either. Delegated so players rendered later are covered, and captured because
// `play` does not bubble.
document.addEventListener("play", (e) => {
  document.querySelectorAll("audio").forEach(a => { if (a !== e.target) a.pause(); });
}, true);

// The frame a ruling key acts on: the lightbox frame if open, else the focused
// tile or card.
function keyTarget() {
  if ($("#lightbox").classList.contains("open")) return CTX[CTX_POS]?.filename;
  if (VIEW === "sheet") return FOCUS;
  return ROWS[SEL]?.filename;
}

document.addEventListener("keydown", (e) => {
  // Any keystroke means the reviewer is working; the blink has served its
  // purpose. Ahead of the guards below so typing a note cancels it too.
  stopBlink();
  // The bulk editor owns the keyboard while it is open.
  if ($("#bulk").classList.contains("open")) {
    if (e.key === "Escape") closeBulk();
    return;
  }
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  // Never shadow a browser shortcut: cmd-c / ctrl-a and friends must still work.
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "e" && RANGE) { e.preventDefault(); return openBulk(); }
  if (e.key === "Escape" && (RANGE || ANCHOR !== null)) return clearRange();
  if (e.key === "u") { e.preventDefault(); return undoFill(); }
  // Fill and the lightbox rulings act on whatever frame is in hand, in any view.
  if (e.key === "A" || e.key === "R") { e.preventDefault(); return fillTo(keyTarget(), e.key === "A" ? "ad" : "content"); }
  // `s` means the same thing wherever a frame is in hand: show it in the sheet.
  if (e.key === "s" && VIEW === "cards" && !$("#lightbox").classList.contains("open")) {
    e.preventDefault();
    return showInSheet(ROWS[SEL]?.filename);
  }
  // While the lightbox is open the arrows walk the broadcast, not the grid.
  if ($("#lightbox").classList.contains("open")) {
    const f = CTX[CTX_POS];
    if (e.key === "Escape") return closeLightbox();
    if (e.key === "ArrowLeft"  || e.key === "k") { e.preventDefault(); return stepTo(CTX_POS - 1); }
    if (e.key === "ArrowRight" || e.key === "j") { e.preventDefault(); return stepTo(CTX_POS + 1); }
    if (e.key === "s") { e.preventDefault(); return openInSheet(); }
    if (e.key === "a") return annotate(f.filename, { verdict: "ad" });
    if (e.key === "r") return annotate(f.filename, { verdict: "content" });
    if (e.key === "o") return annotate(f.filename, { exclude: f.exclude ? null : true });
    if (e.key === "x") return annotate(f.filename, { verdict: null, exclude: null });
    if (e.key === " ") {  // play/pause the clip for the frame on screen
      e.preventDefault();
      const au = $("#lb-audio");
      if (au.src) au.paused ? au.play() : au.pause();
      return;
    }
    return;
  }
  // The sheet moves the same way as the cards, over its own grid. No single
  // rulings here: a thumbnail is too small to rule on, which is what `Enter`
  // is for - it opens the frame where the evidence is. Fill is different: it
  // names the end of a run, and scanning for that is what the sheet is for.
  if (VIEW === "sheet") {
    const at = sheetIndex();
    if (e.key === "j" || e.key === "ArrowRight") { e.preventDefault(); return setSheetSel(at + 1); }
    if (e.key === "k" || e.key === "ArrowLeft") { e.preventDefault(); return setSheetSel(at - 1); }
    if (e.key === "ArrowDown") { e.preventDefault(); return setSheetSel(at + columnCount(".th")); }
    if (e.key === "ArrowUp") { e.preventDefault(); return setSheetSel(at - columnCount(".th")); }
    if (e.key === "Enter" && at >= 0) { e.preventDefault(); return jumpTo(at); }
    return;
  }
  // `r` for racing, not `c`: `c` collided with cmd-c. The modifier guard above
  // now stops that anyway, but the binding is not worth reclaiming.
  const r = ROWS[SEL];
  if (!r) return;
  if (e.key === "a") { setField(SEL, { verdict: "ad" }); setSel(SEL + 1); }
  else if (e.key === "r") { setField(SEL, { verdict: "content" }); setSel(SEL + 1); }
  else if (e.key === "o") setField(SEL, { exclude: r.rec?.exclude ? null : true });
  else if (e.key === "x") setField(SEL, { verdict: null, exclude: null });
  else if (e.key === "n") { e.preventDefault(); document.querySelectorAll(".card")[SEL]?.querySelector(".note")?.focus(); }
  // Arrows move within the grid rather than scrolling it: left/right by one,
  // up/down by a whole row. paint() scrolls whatever lands off-screen.
  else if (e.key === "j" || e.key === "ArrowRight") { e.preventDefault(); setSel(SEL + 1); }
  else if (e.key === "k" || e.key === "ArrowLeft") { e.preventDefault(); setSel(SEL - 1); }
  else if (e.key === "ArrowDown") { e.preventDefault(); setSel(SEL + columnCount(".card")); }
  else if (e.key === "ArrowUp") { e.preventDefault(); setSel(SEL - columnCount(".card")); }
  else if (e.key === "Enter") zoom(r.filename);
});

// Only the backdrop closes — clicks on the image, strip or player must not.
$("#lightbox").onclick = (e) => { if (e.target.id === "lightbox") closeLightbox(); };
$("#lb-close").onclick = closeLightbox;

$("#seledit").onclick = openBulk;
$("#selclear").onclick = clearRange;
$("#bulkcancel").onclick = closeBulk;
$("#bulksave").onclick = saveBulk;
$("#bulk").onclick = (e) => { if (e.target.id === "bulk") closeBulk(); };
document.querySelectorAll("#bulknotemode button").forEach(b =>
  b.onclick = () => { BULK_NOTE = b.dataset.m; paintBulkChoice(); });
// Typing a note is the intent to set one. Default to append where notes already
// exist: it is the one mode that cannot lose what is written there.
$("#bulknote").oninput = () => {
  if ($("#bulknote").value && (BULK_NOTE === "" || BULK_NOTE === CLEAR)) {
    BULK_NOTE = BULK_NOTED ? "append" : "replace";
    paintBulkChoice();
  }
};
$("#broadcast").onchange = (e) => {
  B = e.target.value; PAGE = 1; FOCUS = null; LAST_FILL = null; syncUrl(true); load();
};
document.querySelectorAll("#controls button[data-f]").forEach(b =>
  b.onclick = () => { FILTER = b.dataset.f; PAGE = 1; FOCUS = null; syncUrl(true); load(); });
$("#reload").onclick = async () => { await fetch("/api/reload", { method: "POST" }); await loadBroadcasts(); load(); };
$("#viewtoggle").onclick = () => {
  // Leaving the sheet, carry the focused thumbnail to the cards page holding
  // it - otherwise focusing a frame deep in the sheet and switching would land
  // on page 1 with the focus nowhere in sight.
  if (VIEW === "sheet" && FOCUS) {
    const at = SHEET.findIndex(n => n.filename === FOCUS);
    if (at >= 0) PAGE = Math.floor(at / PER_PAGE) + 1;
  }
  VIEW = VIEW === "sheet" ? "cards" : "sheet";
  BLINK_NEXT = true;
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
  $("#broadcast").value = B;
  $("#size").value = THUMB;
  load();
});

readUrl();
$("#size").value = THUMB;
loadBroadcasts().then(load);
</script>
</body>
</html>
"""


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"record_broadcast.py output root (default {DEFAULT_ROOT})",
    )
    sub = ap.add_subparsers(dest="cmd")
    ap.set_defaults(cmd="serve")

    serve = sub.add_parser("serve", help="run the annotation app (the default)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--broadcast", help="which recording to open first")

    imp = sub.add_parser(
        "import-verdicts",
        help="bring rulings and facets over from the old review app's files",
    )
    imp.add_argument(
        "--broadcast", required=True, help="e.g. tv.youtube.com/USA_4K_Iowa_Corn_350"
    )
    imp.add_argument("--verdicts", type=Path, default=DEFAULT_VERDICTS)
    imp.add_argument("--facets", type=Path, default=DEFAULT_FACETS)
    imp.add_argument("--dataset", default="structure")
    imp.add_argument(
        "--overwrite", action="store_true", help="replace records already present"
    )
    imp.add_argument(
        "--apply", action="store_true", help="write; the default is a dry run"
    )

    # `serve` is the default, but its options are only parsed when it is named,
    # so give a bare invocation the same defaults.
    args = ap.parse_args()
    if args.cmd == "serve" and not hasattr(args, "port"):
        args = ap.parse_args([*sys.argv[1:], "serve"])

    if not args.root.is_dir():
        sys.exit(f"no such directory {args.root}")
    library = scan_library(args.root)
    if not library:
        sys.exit(f"no recordings under {args.root} (looking for */*/{MANIFEST})")

    if args.cmd == "import-verdicts":
        bc = library.get(args.broadcast)
        if bc is None:
            sys.exit(f"unknown broadcast {args.broadcast!r}; have {sorted(library)}")
        stats = import_verdicts(
            bc, args.verdicts, args.facets, args.dataset, args.overwrite, args.apply
        )
        for k, n in sorted(stats.items()):
            print(f"  {k:24s} {n:6d}")
        if args.apply:
            print(f"\nwrote {bc.store.path}")
        else:
            print("\n(dry run — pass --apply to write)")
        return 0

    if args.broadcast and args.broadcast not in library:
        sys.exit(f"unknown broadcast {args.broadcast!r}; have {sorted(library)}")
    for bid, bc in library.items():
        print(f"  {bid:60s} {len(bc.frames()):6d} frames", file=sys.stderr)
    print(f"\n  http://localhost:{args.port}/\n", file=sys.stderr)
    uvicorn.run(
        app_factory(args.root, args.broadcast),
        host=args.host,
        port=args.port,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
