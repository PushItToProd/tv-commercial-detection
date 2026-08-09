"""Migrate a flat save_dir to the images/ + thumbnails/ + audio/ layout.

Frames used to live directly in save_dir, with thumbnails alongside them under a
`compressed_` prefix. They now live in per-type subdirectories so that listing
frames doesn't have to walk the thumbnails and audio too.

    save_dir/2026-01-01T00-00-00_0.jpg             -> save_dir/images/2026-01-01T00-00-00_0.jpg
    save_dir/compressed_2026-01-01T00-00-00_0.jpg  -> save_dir/thumbnails/2026-01-01T00-00-00_0.jpg
    save_dir/2026-01-01T00-00-00_0.wav             -> save_dir/audio/2026-01-01T00-00-00_0.wav

Metadata (labels.json, features.jsonl, classifications.jsonl) stays put; the
records inside reference bare filenames, so nothing in them needs rewriting.

Usage:
    uv run python scripts/migrate_frames_to_subdirs.py            # dry run
    uv run python scripts/migrate_frames_to_subdirs.py --apply
"""

import argparse
import os
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
AUDIO_SUFFIXES = {".wav"}
THUMBNAIL_PREFIX = "compressed_"


def plan_moves(save_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each loose media file at the save_dir root with its new home."""
    moves: list[tuple[Path, Path]] = []
    with os.scandir(save_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                if name.startswith(THUMBNAIL_PREFIX):
                    dest = save_dir / "thumbnails" / name[len(THUMBNAIL_PREFIX) :]
                else:
                    dest = save_dir / "images" / name
            elif suffix in AUDIO_SUFFIXES:
                dest = save_dir / "audio" / name
            else:
                continue  # metadata and anything else stays at the root
            moves.append((save_dir / name, dest))
    return moves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d", "--save-dir",
        type=Path,
        default=Path(__file__).parent.parent / "frames",
        help="Frame save directory (default: server/frames)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this the script only reports.",
    )
    args = parser.parse_args()

    save_dir: Path = args.save_dir
    if not save_dir.is_dir():
        sys.exit(f"Error: '{save_dir}' is not a directory")

    moves = plan_moves(save_dir)
    if not moves:
        print(f"Nothing to migrate — no loose media files in {save_dir}")
        return 0

    by_dest_dir: dict[str, int] = {}
    for _, dest in moves:
        by_dest_dir[dest.parent.name] = by_dest_dir.get(dest.parent.name, 0) + 1
    for dest_dir, count in sorted(by_dest_dir.items()):
        print(f"  {count:>7,} -> {save_dir / dest_dir}/")

    if not args.apply:
        print(f"\nDry run: {len(moves):,} files would move. Re-run with --apply.")
        return 0

    for dest_dir in by_dest_dir:
        (save_dir / dest_dir).mkdir(parents=True, exist_ok=True)

    moved = 0
    skipped: list[Path] = []
    for src, dest in moves:
        if dest.exists():
            skipped.append(src)
            continue
        # Same filesystem, so this is an atomic rename rather than a copy.
        src.rename(dest)
        moved += 1

    print(f"\nMoved {moved:,} files.")
    if skipped:
        print(
            f"Skipped {len(skipped):,} whose destination already existed "
            f"(e.g. {skipped[0].name}) — left in place for you to review."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
