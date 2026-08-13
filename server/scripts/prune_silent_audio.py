"""Delete silent audio clips from a save dir.

When the native host monitors a sink the browser isn't playing to, the clips it
returns are digital silence — right length, right cadence, nothing in them.
Whole stretches of the corpus can be dead this way before anyone notices, and
they cost disk and mislead anything trained on the audio.

A clip is silent if its peak amplitude falls at or below --threshold, the same
measure `audio_health` uses to warn about live capture. A genuinely muted or
paused broadcast reads the same way, which is fine: a silent clip carries no
signal either way, so nothing is lost by dropping it.

Only `audio/` is touched. Clips share a stem with their frame but carry no
records of their own — labels, features and classifications all key on the
image — so removing one dangles nothing and the frame stays put.

Clips that can't be parsed as WAV are reported and left alone, on the same
reasoning as in `audio_health`: an unreadable clip says nothing about whether
capture was live.

Usage:
    uv run python scripts/prune_silent_audio.py            # dry run + per-day report
    uv run python scripts/prune_silent_audio.py --apply
"""

import argparse
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tv_commercial_detector.audio_health import peak_amplitude  # noqa: E402
from tv_commercial_detector.config import app_config  # noqa: E402

AUDIO_SUFFIXES = {".wav"}
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def measure_one(path: Path):
    """(name, size, peak) — peak is None if the clip can't be read."""
    try:
        data = path.read_bytes()
        size = len(data)
    except OSError as e:
        print(f"  skipping {path.name}: {e}", file=sys.stderr)
        return None
    return path.name, size, peak_amplitude(data)


def measure_all(paths: list[Path], workers: int):
    rows = []
    # The 3.14 default start method (forkserver) needs an AF_UNIX socket that
    # sandboxed environments deny; fork does not.
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, row in enumerate(pool.map(measure_one, paths, chunksize=32), 1):
            if row is not None:
                rows.append(row)
            if i % 5000 == 0:
                print(f"  measured {i:,}/{len(paths):,}")
    return rows


def day_of(name: str) -> str:
    """Capture date from the filename; both naming conventions lead with it."""
    match = DATE_RE.match(name)
    return match.group(1) if match else "unknown"


def human(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def report_by_day(rows, threshold: float) -> None:
    """Per-day silent counts — dead capture starts on a day and stays."""
    clips: dict[str, int] = defaultdict(int)
    silent: dict[str, int] = defaultdict(int)
    silent_bytes: dict[str, int] = defaultdict(int)

    for name, size, peak in rows:
        day = day_of(name)
        clips[day] += 1
        if peak is not None and peak <= threshold:
            silent[day] += 1
            silent_bytes[day] += size

    print(f"\n  {'date':<12}{'clips':>8}{'silent':>9}{'%':>6}{'silent size':>14}")
    for day in sorted(clips):
        pct = 100 * silent[day] / clips[day]
        print(
            f"  {day:<12}{clips[day]:>8,}{silent[day]:>9,}{pct:>5.0f}%"
            f"{human(silent_bytes[day]):>14}"
        )
    total_clips = sum(clips.values())
    total_silent = sum(silent.values())
    pct = 100 * total_silent / total_clips if total_clips else 0
    print(
        f"  {'total':<12}{total_clips:>8,}{total_silent:>9,}{pct:>5.0f}%"
        f"{human(sum(silent_bytes.values())):>14}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--save-dir",
        type=Path,
        default=Path(__file__).parent.parent / "frames",
        help="Frame save directory (default: server/frames)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=app_config.audio_silence_threshold,
        help="Peak amplitude, as a fraction of full scale, at or below which a "
        f"clip counts as silent (default: {app_config.audio_silence_threshold})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this the script only reports.",
    )
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    audio_dir: Path = args.save_dir / "audio"
    if not audio_dir.is_dir():
        sys.exit(f"Error: '{audio_dir}' is not a directory")

    paths = [p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES]
    if not paths:
        print(f"No audio clips in {audio_dir}.")
        return 0

    print(
        f"Measuring {len(paths):,} clips in {audio_dir} with {args.workers} workers..."
    )
    rows = measure_all(paths, args.workers)

    report_by_day(rows, args.threshold)

    doomed = [
        (n, s) for n, s, peak in rows if peak is not None and peak <= args.threshold
    ]
    unreadable = [n for n, _s, peak in rows if peak is None]
    if unreadable:
        print(f"\nUnreadable: {len(unreadable):,} clip(s), left alone")

    if not doomed:
        print("\nNo silent clips found.")
        return 0

    print(
        f"\nRemoving {len(doomed):,} of {len(rows):,} clips "
        f"({100 * len(doomed) / len(rows):.0f}%), {human(sum(s for _n, s in doomed))}"
    )

    if not args.apply:
        print("\nDry run: nothing deleted. Re-run with --apply.")
        return 0

    for name, _size in doomed:
        (audio_dir / name).unlink(missing_ok=True)
    print(f"Deleted {len(doomed):,} silent clips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
