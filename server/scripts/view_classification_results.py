#!/usr/bin/env python3
"""Standalone FastAPI viewer for check_classification.py JSONL output.

Usage:
    uv run python view_results.py classification_results14c.jsonl
    uv run python view_results.py path/to/results.jsonl --frames-dir path/to/frames --port 8765
"""

import argparse
import json
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View check_classification.py JSONL results in a browser"
    )
    parser.add_argument("jsonl_file", help="Path to the JSONL results file")
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory containing frame images (default: 'frames/' next to the JSONL file)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(jsonl_path: Path) -> tuple[list[dict], dict | None]:
    frames: list[dict] = []
    summary: dict | None = None
    pending: list[str] = []
    depth = 0

    def process(text: str) -> None:
        nonlocal summary
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return
        if obj.get("status") == "summary":
            summary = obj
        else:
            frames.append(obj)

    with jsonl_path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            depth += line.count("{") - line.count("}")
            pending.append(line)
            if depth <= 0:
                process(" ".join(pending))
                pending.clear()
                depth = 0
    return frames, summary


# ── HTML template ─────────────────────────────────────────────────────────────
# Placeholders replaced at runtime to avoid escaping all JS braces.

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Results — __TITLE__</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body { font-family: sans-serif; background: #1a1a1a; color: #eee; margin: 0; padding: 1rem; }
  h1 { margin: 0 0 0.75rem; font-size: 1.1rem; opacity: 0.6; word-break: break-all; }

  /* Summary bar */
  .summary {
    background: #252525; border-radius: 8px; padding: 0.75rem 1rem;
    display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem;
    margin-bottom: 1rem; align-items: center;
  }
  .stat { display: flex; flex-direction: column; align-items: center; min-width: 4rem; }
  .stat .val { font-size: 1.4rem; font-weight: bold; }
  .stat .lbl { font-size: 0.68rem; opacity: 0.5; text-transform: uppercase; letter-spacing: .03em; }
  .stat.bad  .val { color: #f55; }
  .stat.good .val { color: #4d4; }
  .stat.timing .val { font-size: 0.85rem; color: #999; }
  .divider { width: 1px; background: #383838; align-self: stretch; }

  /* Filter controls */
  #controls {
    margin-bottom: 0.75rem; display: flex; flex-wrap: wrap;
    gap: 0.4rem; align-items: center;
  }
  .filter-btn {
    padding: 0.28rem 0.75rem; border: 2px solid #444; border-radius: 4px;
    background: transparent; color: #ccc; cursor: pointer; font-size: 0.82rem;
    transition: background .12s, border-color .12s;
  }
  .filter-btn:hover  { background: #2e2e2e; }
  .filter-btn.active { background: #383838; border-color: #777; color: #fff; }
  #counter { margin-left: auto; opacity: 0.45; font-size: 0.82rem; }

  /* Cards */
  .grid { display: flex; flex-wrap: wrap; gap: 0.75rem; }
  .card {
    background: #242424; border-radius: 8px; overflow: hidden;
    width: 260px; display: flex; flex-direction: column;
    border: 2px solid #3a3a3a;
  }
  .card.incorrect { border-color: #c60; }
  .card.unlabeled { border-color: #334; }
  .card.ignored   { border-color: #2a2a2a; opacity: 0.45; }

  .card img {
    width: 100%; display: block; cursor: zoom-in;
    background: #111; min-height: 80px;
  }
  .card .filename {
    padding: 0.28rem 0.5rem; font-size: 0.63rem; opacity: 0.38;
    word-break: break-all; border-bottom: 1px solid #2e2e2e;
  }
  .card .labels { padding: 0.4rem 0.5rem; display: flex; flex-direction: column; gap: 0.25rem; }
  .label-row { display: flex; gap: 0.4rem; align-items: center; font-size: 0.78rem; }
  .lk { opacity: 0.45; width: 4.5rem; flex-shrink: 0; }

  /* Badges */
  .badge {
    display: inline-block; padding: 0.1rem 0.45rem;
    border-radius: 3px; font-size: 0.72rem; font-weight: bold;
  }
  .badge.ad         { background: #a02; color: #fff; }
  .badge.content    { background: #060; color: #dfd; }
  .badge.unknown    { background: #555; color: #ccc; }
  .badge.correct    { background: #0a2e0a; color: #5e5; }
  .badge.incorrect  { background: #3a1800; color: #f90; }
  .badge.unlabeled  { background: #1a1a30; color: #77a; }
  .badge.ignored    { background: #1e1e1e; color: #555; }

  .elapsed { opacity: 0.35; font-size: 0.68rem; margin-top: 0.2rem; }

  /* Model reply */
  .model-reply {
    margin: 0.3rem 0.5rem 0.5rem;
    padding: 0.4rem 0.5rem;
    background: #161616; border-radius: 4px; border-left: 2px solid #c60;
    font-size: 0.68rem; line-height: 1.5; color: #bbb;
    max-height: 7rem; overflow-y: auto;
  }

  /* Lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0;
    background: rgba(0,0,0,.9);
    align-items: center; justify-content: center; z-index: 100;
  }
  #lightbox.open { display: flex; }
  #lightbox img { max-width: 95vw; max-height: 93vh; border-radius: 6px; }
  #lb-close {
    position: fixed; top: 0.6rem; right: 1rem;
    font-size: 2.5rem; cursor: pointer; color: #fff;
    line-height: 1; user-select: none;
  }
</style>
</head>
<body>
<h1>__TITLE__</h1>

<div class="summary" id="summary-bar"></div>

<div id="controls">
  <button class="filter-btn active" data-filter="all">All</button>
  <button class="filter-btn" data-filter="incorrect">Incorrect</button>
  <button class="filter-btn" data-filter="correct">Correct</button>
  <button class="filter-btn" data-filter="unlabeled">Unlabeled</button>
  <button class="filter-btn" data-filter="ignored">Ignored</button>
  <span id="counter"></span>
</div>

<div class="grid" id="grid"></div>

<div id="lightbox">
  <span id="lb-close" title="Close">&times;</span>
  <img id="lb-img" src="" alt="">
</div>

<script>
const FRAMES  = __FRAMES_JSON__;
const SUMMARY = __SUMMARY_JSON__;

// ── Summary bar ───────────────────────────────────────────────────────────────
function buildSummary() {
  const bar = document.getElementById('summary-bar');
  if (!SUMMARY) { bar.textContent = 'No summary available.'; return; }

  const pct = (SUMMARY.incorrect_pct * 100).toFixed(1);
  const stats = [
    { lbl: 'Total',     val: SUMMARY.total,                                 cls: '' },
    { lbl: 'Correct',   val: SUMMARY.total - SUMMARY.incorrect,             cls: 'good' },
    { lbl: 'Incorrect', val: `${SUMMARY.incorrect} (${pct}%)`,              cls: SUMMARY.incorrect > 0 ? 'bad' : 'good' },
  ];
  if (SUMMARY.unlabeled) stats.push({ lbl: 'Unlabeled', val: SUMMARY.unlabeled, cls: '' });
  if (SUMMARY.ignored)   stats.push({ lbl: 'Ignored',   val: SUMMARY.ignored,   cls: '' });

  stats.forEach(({ lbl, val, cls }, i) => {
    if (i > 0) bar.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
    bar.insertAdjacentHTML('beforeend',
      `<div class="stat ${cls}"><span class="val">${val}</span><span class="lbl">${lbl}</span></div>`);
  });

  if (SUMMARY.avg_elapsed != null) {
    bar.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
    bar.insertAdjacentHTML('beforeend',
      `<div class="stat timing">
         <span class="val">avg ${SUMMARY.avg_elapsed}s &middot; med ${SUMMARY.median_elapsed}s &middot; min ${SUMMARY.min_elapsed}s &middot; max ${SUMMARY.max_elapsed}s &middot; total ${SUMMARY.total_elapsed}s</span>
         <span class="lbl">Timing</span>
       </div>`);
  }

  if (SUMMARY.incorrectly_marked_as_ads?.length || SUMMARY.incorrectly_marked_as_content?.length || SUMMARY.incorrectly_unknown?.length) {
    bar.insertAdjacentHTML('beforeend', '<div class="divider"></div>');
    const parts = [];
    if (SUMMARY.incorrectly_marked_as_ads?.length)     parts.push(`${SUMMARY.incorrectly_marked_as_ads.length} false-ad`);
    if (SUMMARY.incorrectly_marked_as_content?.length) parts.push(`${SUMMARY.incorrectly_marked_as_content.length} false-content`);
    if (SUMMARY.incorrectly_unknown?.length)           parts.push(`${SUMMARY.incorrectly_unknown.length} unknown`);
    bar.insertAdjacentHTML('beforeend',
      `<div class="stat bad"><span class="val">${parts.join(' · ')}</span><span class="lbl">Error breakdown</span></div>`);
  }
}

// ── Grid ──────────────────────────────────────────────────────────────────────
function buildGrid() {
  const grid = document.getElementById('grid');
  FRAMES.forEach(frame => {
    const card = document.createElement('div');
    card.className = 'card ' + frame.status;
    card.dataset.status = frame.status;

    if (frame.file) {
      const img = document.createElement('img');
      img.src = '/frames/' + encodeURIComponent(frame.file);
      img.loading = 'lazy';
      img.title = 'Click to enlarge';
      img.addEventListener('click', () => openLightbox(img.src));
      card.appendChild(img);

      const fn = document.createElement('div');
      fn.className = 'filename';
      fn.textContent = frame.file;
      card.appendChild(fn);
    }

    const labels = document.createElement('div');
    labels.className = 'labels';

    labels.insertAdjacentHTML('beforeend',
      `<div class="label-row">
         <span class="lk">Status</span>
         <span class="badge ${frame.status}">${frame.status.toUpperCase()}</span>
       </div>`);

    if (frame.expected != null) {
      labels.insertAdjacentHTML('beforeend',
        `<div class="label-row">
           <span class="lk">Expected</span>
           <span class="badge ${frame.expected}">${frame.expected.toUpperCase()}</span>
         </div>`);
    }
    if (frame.classified != null) {
      labels.insertAdjacentHTML('beforeend',
        `<div class="label-row">
           <span class="lk">Got</span>
           <span class="badge ${frame.classified}">${frame.classified.toUpperCase()}</span>
         </div>`);
    }
    if (frame.elapsed != null) {
      labels.insertAdjacentHTML('beforeend', `<div class="elapsed">${frame.elapsed}s</div>`);
    }
    card.appendChild(labels);

    if (frame.model_reply) {
      const reply = document.createElement('div');
      reply.className = 'model-reply';
      if (typeof frame.model_reply === 'object') {
        reply.textContent = `type=${frame.model_reply.type} (${frame.model_reply.reason})` + (!frame.model_reply.reply ? '' : ` -- ${frame.model_reply.reply}`);
      } else {
        reply.textContent = JSON.stringify(frame.model_reply);
      }
      card.appendChild(reply);
    }

    grid.appendChild(card);
  });
}

// ── Filter ────────────────────────────────────────────────────────────────────
let currentFilter = 'all';

function applyFilter() {
  const cards = [...document.querySelectorAll('.card')];
  cards.forEach(c => {
    c.style.display = (currentFilter === 'all' || c.dataset.status === currentFilter) ? '' : 'none';
  });
  const visible = cards.filter(c => c.style.display !== 'none').length;
  document.getElementById('counter').textContent = `${visible} / ${cards.length} shown`;
}

document.querySelectorAll('.filter-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.filter;
    applyFilter();
  });
});

// ── Lightbox ──────────────────────────────────────────────────────────────────
function openLightbox(src) {
  document.getElementById('lb-img').src = src;
  document.getElementById('lightbox').classList.add('open');
}
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target === e.currentTarget || e.target.id === 'lb-close')
    e.currentTarget.classList.remove('open');
});

// ── Init ──────────────────────────────────────────────────────────────────────
buildSummary();
buildGrid();
applyFilter();
</script>
</body>
</html>
"""


def render_html(title: str, frames: list[dict], summary: dict | None) -> str:
    return (
        _HTML
        .replace("__TITLE__", title)
        .replace("__FRAMES_JSON__", json.dumps(frames))
        .replace("__SUMMARY_JSON__", json.dumps(summary))
    )


# ── FastAPI app ───────────────────────────────────────────────────────────────

def create_app(title: str, frames: list[dict], summary: dict | None, frames_dir: Path) -> FastAPI:
    app = FastAPI()
    _frames_dir = frames_dir.resolve()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(render_html(title, frames, summary))

    @app.get("/frames/{filename}")
    def serve_frame(filename: str) -> FileResponse:
        # Resolve and validate to prevent path traversal
        target = (_frames_dir / filename).resolve()
        if not target.is_relative_to(_frames_dir) or not target.is_file():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(target)

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl_file).resolve()

    if not jsonl_path.is_file():
        sys.exit(f"Error: '{jsonl_path}' not found")

    frames_dir = (
        Path(args.frames_dir).resolve()
        if args.frames_dir
        else (jsonl_path.parent / "frames").resolve()
    )
    if not frames_dir.is_dir():
        print(
            f"Warning: frames directory '{frames_dir}' not found — images won't load",
            file=sys.stderr,
        )

    frames, summary = load_results(jsonl_path)
    title = jsonl_path.name

    print(f"Loaded {len(frames)} frame entries from {title}")
    if summary:
        incorrect = summary.get("incorrect", 0)
        total = summary.get("total", 0)
        pct = summary.get("incorrect_pct", 0) * 100
        print(f"Summary: {total} classified, {incorrect} incorrect ({pct:.1f}%)")

    app = create_app(title, frames, summary, frames_dir)
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
