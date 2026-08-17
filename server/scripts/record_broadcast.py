#!/usr/bin/env python3
"""Standalone receiver that archives a whole broadcast to disk.

Speaks the same `/receive` and `/video-state` protocol as the detector server,
but does no classification, no matrix switching and no debouncing — every frame
(and its audio clip, if the extension sends one) is written straight to disk.

The browser extension posts to every endpoint in its list, so this can run
alongside the real detector: add `http://<host>:11680/receive` as a second
endpoint in the extension popup and both get the same frames.

    uv run python scripts/record_broadcast.py -d /mnt/data/tv-commercial-detector/full_broadcast_frames

Frames land in a per-broadcast directory derived from the page's hostname, the
network name (YouTube TV reports one; other sites don't) and the video or page
title:

    <out-dir>/tv.youtube.com/Oregon-s_FOX_Autotrader_400/
    <out-dir>/play.hbomax.com/eero_400_-_HBO_Max/

Each of those uses the same layout as the detector's save dir — `images/`,
`audio/` and a `classifications.jsonl` keyed on bare filenames — so a recording
can be pointed at directly with `DETECTOR_SAVE_DIR` for review, or fed to
`dedupe_frames.py` and the other scripts. Filenames follow the frame saver's
convention so `frame_timestamp()` can order them.

A broadcast that changes title mid-stream (a pre-race show rolling into the
race) starts a new directory; going back to a title already seen resumes
appending to its directory.

Each record also carries the player's timebase — `video_id`, `video_duration`,
`is_live`, `seekable_start`, `seekable_end` — which is what says whether
`video_offset` is a position in the program or just time since the player
loaded. See `video_timebase.py`; the parsing is shared with the detector's
`/receive` so both receivers write the same thing.
"""

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, UploadFile

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tv_commercial_detector.video_timebase import parse_timebase  # noqa: E402

DEFAULT_PORT = 11680

# Characters that survive sanitization as-is. Everything else is either a
# separator (whitespace becomes "_") or noise (anything else becomes "-").
SAFE_CHARS = re.compile(r"[A-Za-z0-9._-]")


def sanitize(text: str) -> str:
    """Turn a page/video/network title into a single safe path component.

    Format and control characters are dropped outright — YouTube TV and HBO Max
    both wrap titles in directional isolates (U+2068/U+2069), which are
    invisible in the UI but would otherwise end up in the directory name.
    Whitespace (including the non-breaking space in "HBO Max") becomes "_", and
    every other character becomes "-", so `\u2068eero 400\u2069 • HBO Max` reads
    as `eero_400_-_HBO_Max`.
    """
    cleaned = []
    for ch in text:
        if unicodedata.category(ch) in ("Cf", "Cc"):
            continue
        if ch.isspace():
            cleaned.append("_")
        elif SAFE_CHARS.match(ch):
            cleaned.append(ch)
        else:
            cleaned.append("-")
    out = "".join(cleaned)
    out = re.sub(r"-{2,}", "-", out)
    out = re.sub(r"_{2,}", "_", out)
    return out.strip("-_.")


def hostname_for(page_url: str) -> str:
    host = urlparse(page_url).hostname if page_url else None
    return sanitize(host) if host else "unknown-host"


def broadcast_slug(page_title: str, video_title: str, network_name: str) -> str:
    """Directory name for one broadcast.

    The video title is the specific one ("Autotrader 400"); the page title is
    the fallback for sites that don't report a video title, where it's the only
    thing naming the broadcast. The network name goes in front because YouTube
    TV's page title is usually boilerplate ("Home - YouTube TV") and the network
    is what distinguishes two simultaneous feeds.
    """
    parts = [sanitize(network_name), sanitize(video_title or page_title)]
    slug = "_".join(p for p in parts if p)
    return slug or "unknown-broadcast"


@dataclass
class Session:
    """One broadcast directory and what's been written to it."""

    root: Path
    frames: int = 0
    clips: int = 0
    image_bytes: int = 0
    audio_bytes: int = 0
    started: datetime = field(default_factory=datetime.now)

    def prepare(self) -> None:
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        (self.root / "audio").mkdir(parents=True, exist_ok=True)


class Recorder:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.sessions: dict[Path, Session] = {}
        self.current: Session | None = None
        self.skipped = 0
        # Serializes the append to classifications.jsonl; without it two
        # overlapping requests can interleave partial lines.
        self.lock = asyncio.Lock()

    def session_for(self, root: Path) -> Session:
        session = self.sessions.get(root)
        if session is None:
            session = Session(root=root)
            session.prepare()
            self.sessions[root] = session
            log(f"recording to {root}")
        elif session is not self.current:
            log(f"back to {root}")
        self.current = session
        return session

    def write(
        self,
        session: Session,
        filename: str,
        frame: bytes,
        audio: bytes | None,
        record: dict,
    ) -> None:
        """Blocking disk work for one frame; called via asyncio.to_thread."""
        (session.root / "images" / filename).write_bytes(frame)
        session.frames += 1
        session.image_bytes += len(frame)
        if audio is not None:
            stem = Path(filename).stem
            (session.root / "audio" / f"{stem}.wav").write_bytes(audio)
            session.clips += 1
            session.audio_bytes += len(audio)
        with (session.root / "classifications.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    def summary(self) -> str:
        if not self.sessions:
            return "Nothing recorded."
        lines = []
        for root, s in self.sessions.items():
            elapsed = datetime.now() - s.started
            lines.append(
                f"  {root}\n"
                f"    {s.frames} frames ({human(s.image_bytes)}),"
                f" {s.clips} audio clips ({human(s.audio_bytes)}),"
                f" over {str(elapsed).split('.')[0]}"
            )
        if self.skipped:
            lines.append(
                f"  ({self.skipped} requests carried no image"
                f" — paused, seeking, or no video on the page)"
            )
        return "Recorded:\n" + "\n".join(lines)


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def as_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes")


def build_app(recorder: Recorder) -> FastAPI:
    router = APIRouter()

    @router.post("/receive")
    async def receive(
        image: UploadFile | None = File(default=None),
        audio: UploadFile | None = File(default=None),
        is_paused: str = Form(default=""),
        is_seeking: str = Form(default=""),
        timestamp: str = Form(default=""),
        page_title: str = Form(default=""),
        page_url: str = Form(default=""),
        video_title: str = Form(default=""),
        network_name: str = Form(default=""),
        video_offset: str = Form(default=""),
        video_id: str = Form(default=""),
        video_duration: str = Form(default=""),
        seekable_start: str = Form(default=""),
        seekable_end: str = Form(default=""),
    ):
        if image is None:
            # The extension skips the screenshot while paused or seeking, or
            # when it can't find a player element on the page at all.
            recorder.skipped += 1
            return {"recorded": False, "reason": "no image"}

        ext = Path(image.filename).suffix.lower() if image.filename else ".jpg"
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"

        frame_bytes = await image.read()
        audio_bytes = await audio.read() if audio is not None else None

        # The extension stamps each capture; fall back to arrival time.
        captured = datetime.now().isoformat()
        if timestamp:
            try:
                captured = datetime.fromisoformat(timestamp).isoformat()
            except ValueError:
                pass
        filename = captured.replace(":", "-").replace(".", "-") + ext

        root = (
            recorder.out_dir
            / hostname_for(page_url)
            / broadcast_slug(page_title, video_title, network_name)
        )
        offset = float(video_offset) if video_offset else None
        timebase = parse_timebase(
            video_id, video_duration, seekable_start, seekable_end
        )
        record = {
            "filename": filename,
            "timestamp": captured,
            "save_reason": "full_broadcast",
            "page_title": page_title,
            "page_url": page_url,
            "video_title": video_title,
            "network_name": network_name,
            "video_offset": offset,
            **timebase.as_record(),
            "is_paused": as_bool(is_paused),
            "is_seeking": as_bool(is_seeking),
            "has_audio": audio_bytes is not None,
        }

        async with recorder.lock:
            session = recorder.session_for(root)
            await asyncio.to_thread(
                recorder.write, session, filename, frame_bytes, audio_bytes, record
            )

        offset_str = f"{offset:.1f}s" if offset is not None else "?"
        audio_str = f" + {human(len(audio_bytes))} audio" if audio_bytes else ""
        log(
            f"{root.name} #{session.frames}  {filename}"
            f"  {human(len(frame_bytes))}{audio_str}  offset {offset_str}"
        )
        return {"recorded": True, "filename": filename}

    @router.post("/video-state")
    async def video_state(
        is_paused: str = Form(default=""),
        is_seeking: str = Form(default=""),
        no_video: str = Form(default=""),
        page_title: str = Form(default=""),
        video_title: str = Form(default=""),
    ):
        status = (
            "no video on page"
            if as_bool(no_video)
            else "paused"
            if as_bool(is_paused)
            else "seeking"
            if as_bool(is_seeking)
            else "resumed"
        )
        log(f"video {status}  |  {video_title or page_title}")
        return {"recorded": False}

    @router.get("/status")
    async def status():
        return {
            "out_dir": str(recorder.out_dir),
            "current": str(recorder.current.root) if recorder.current else None,
            "sessions": [
                {
                    "dir": str(root),
                    "frames": s.frames,
                    "audio_clips": s.clips,
                    "image_bytes": s.image_bytes,
                    "audio_bytes": s.audio_bytes,
                    "started": s.started.isoformat(),
                }
                for root, s in recorder.sessions.items()
            ],
            "skipped": recorder.skipped,
        }

    app = FastAPI(title="Broadcast recorder")
    app.include_router(router)
    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Receive and archive an entire broadcast's frames and audio.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=True,
        type=Path,
        help="output directory; per-broadcast subdirectories are created under it",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    out_dir = args.dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    recorder = Recorder(out_dir)
    app = build_app(recorder)

    log(f"Recording to {out_dir}")
    log(
        f"Point the extension at http://<this-host>:{args.port}/receive — Ctrl-C to stop"
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    print()
    print(recorder.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
