"""Remove near-duplicate frames from a save dir.

Bursts of saved frames and repeated commercials leave the save dir full of
visually interchangeable images. This groups them by perceptual hash and keeps
one representative of each group.

Two properties keep it from eating the corpus (see notes/frame-deduplication.md
for the measurements behind them):

- Grouping is greedy keep-first, not transitive. A frame is dropped only if it
  is within the threshold of a frame already kept, so every removal is
  justified against the survivor that replaces it. Connected components over
  the same relation chain unrelated frames together — at threshold 16 a single
  component covers 99% of the corpus.
- Frames carrying manual work (a label or a feature record) are never removed,
  and are preferred as representatives.

Audio is left alone unless --include-audio is passed: a clip shares only its
stem with its frame, so a static shot with different commentary is exactly the
case where the image is redundant and the audio is not.

Usage:
    uv run python scripts/dedupe_frames.py                     # dry run
    uv run python scripts/dedupe_frames.py --report-conflicts  # audit labels only
    uv run python scripts/dedupe_frames.py --apply
    uv run python scripts/dedupe_frames.py --apply --include-audio --drop-blank
"""

import argparse
import json
import multiprocessing
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tv_commercial_detector.routes.review import frame_sort_key  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_THRESHOLD = 10
# dhash is allowed a little more slack than phash: it is the corroborating
# signal, not the primary one, and holding it to the same bound rejects genuine
# duplicates that differ by a scoreboard ticker.
DHASH_SLACK = 4
# Solid-black fade transitions. Every near-uniform frame lands near zero, which
# is what lets unrelated clusters bridge through them.
BLANK_HASHES = {0x0000000000000000, 0x0000000000000080}


def hash_one(path: Path):
    """(name, size, phash, dhash) as plain ints, or None if unreadable."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            ph = int(str(imagehash.phash(im)), 16)
            dh = int(str(imagehash.dhash(im)), 16)
    except Exception as e:
        print(f"  skipping {path.name}: {e}", file=sys.stderr)
        return None
    return path.name, path.stat().st_size, ph, dh


def hash_all(paths: list[Path], workers: int):
    rows = []
    # The 3.14 default start method (forkserver) needs an AF_UNIX socket that
    # sandboxed environments deny; fork does not.
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, row in enumerate(pool.map(hash_one, paths, chunksize=64), 1):
            if row is not None:
                rows.append(row)
            if i % 10000 == 0:
                print(f"  hashed {i:,}/{len(paths):,}")
    return rows


def load_protected(save_dir: Path) -> set[str]:
    """Filenames carrying manual work, which must survive."""
    protected: set[str] = set()
    labels_path = save_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        protected.update(name for name, value in labels.items() if value is not None)
    features_path = save_dir / "features.jsonl"
    if features_path.exists():
        for line in features_path.read_text().splitlines():
            if line.strip():
                protected.add(json.loads(line)["filename"])
    return protected


def load_labels(save_dir: Path) -> dict[str, str]:
    path = save_dir / "labels.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text()).items() if v is not None}


def select_duplicates(rows, threshold: int, protected: set[str]):
    """Greedy keep-first pass. Returns (drops, kept_count).

    drops is a list of (dropped_name, representative_name, distance).
    """
    # Protected frames first so they become representatives, then chronological
    # so the earliest frame of a burst is the one that survives.
    rows = sorted(rows, key=lambda r: (r[0] not in protected, frame_sort_key(r[0])))

    phashes = np.empty(len(rows), dtype=np.uint64)
    kept_names: list[str] = []
    kept_dhash: list[int] = []
    n_kept = 0
    drops: list[tuple[str, str, int]] = []

    for name, _size, ph, dh in rows:
        if n_kept:
            dist = np.bitwise_count(np.bitwise_xor(np.uint64(ph), phashes[:n_kept]))
            j = int(np.argmin(dist))
            if dist[j] <= threshold and name not in protected:
                if int(dh ^ kept_dhash[j]).bit_count() <= threshold + DHASH_SLACK:
                    drops.append((name, kept_names[j], int(dist[j])))
                    continue
        phashes[n_kept] = ph
        kept_names.append(name)
        kept_dhash.append(dh)
        n_kept += 1

    return drops, n_kept


def find_conflicts(rows, threshold: int, labels: dict[str, str]):
    """Pairs within threshold whose stored labels disagree."""
    labelled = [r for r in rows if r[0] in labels]
    if len(labelled) < 2:
        return []
    ph = np.array([r[2] for r in labelled], dtype=np.uint64)
    dist = np.bitwise_count(np.bitwise_xor(ph[:, None], ph[None, :]))
    rows_i, cols_i = np.triu_indices(len(labelled), 1)
    close = dist[rows_i, cols_i] <= threshold
    conflicts = []
    for a, b in zip(rows_i[close], cols_i[close]):
        na, nb = labelled[a][0], labelled[b][0]
        if labels[na] != labels[nb]:
            conflicts.append((na, labels[na], nb, labels[nb], int(dist[a, b])))
    return sorted(conflicts, key=lambda c: c[4])


def human(n: int) -> str:
    return f"{n / 2**30:.2f} GiB" if n >= 2**30 else f"{n / 2**20:.0f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-d", "--save-dir",
        type=Path,
        default=Path(__file__).parent.parent / "frames",
        help="Frame save directory (default: server/frames)",
    )
    parser.add_argument(
        "-t", "--threshold",
        type=int,
        default=DEFAULT_THRESHOLD,
        help=f"Max perceptual hash distance to treat as duplicate (default: {DEFAULT_THRESHOLD}). "
             "Values above 11 merge frames with different manual labels.",
    )
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Also delete the audio clip belonging to each removed frame. Off by "
             "default: identical images can carry different commentary.",
    )
    parser.add_argument(
        "--drop-blank",
        action="store_true",
        help="Also delete solid-black transition frames outright.",
    )
    parser.add_argument(
        "--report-conflicts",
        action="store_true",
        help="Report near-duplicate frames with disagreeing labels, then exit.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files. Without this the script only reports.",
    )
    parser.add_argument("-j", "--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    save_dir: Path = args.save_dir
    images_dir = save_dir / "images"
    if not images_dir.is_dir():
        sys.exit(f"Error: '{images_dir}' is not a directory")
    if args.threshold > 11 and not args.report_conflicts:
        print(f"Warning: threshold {args.threshold} exceeds 11, where frames with "
              f"differing manual labels start being merged.\n", file=sys.stderr)

    paths = [p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    print(f"Hashing {len(paths):,} images with {args.workers} workers...")
    rows = hash_all(paths, args.workers)

    if args.report_conflicts:
        labels = load_labels(save_dir)
        conflicts = find_conflicts(rows, args.threshold, labels)
        if not conflicts:
            print(f"\nNo label conflicts within distance {args.threshold}.")
            return 0
        print(f"\n{len(conflicts)} conflicting pair(s) within distance {args.threshold}:")
        for na, la, nb, lb, dist in conflicts:
            print(f"  d={dist:<3} {na} [{la}]  vs  {nb} [{lb}]")
        return 0

    protected = load_protected(save_dir)
    print(f"Protected (labelled or annotated): {len(protected):,}")

    drops, n_kept = select_duplicates(rows, args.threshold, protected)
    doomed = {name for name, _rep, _d in drops}

    if args.drop_blank:
        blank = {n for n, _s, ph, _dh in rows if ph in BLANK_HASHES and n not in protected}
        new_blank = blank - doomed
        doomed |= blank
        print(f"Blank frames: {len(blank):,} ({len(new_blank):,} not already duplicates)")

    sizes = {name: size for name, size, _ph, _dh in rows}
    image_bytes = sum(sizes[n] for n in doomed)

    thumbs_dir, audio_dir = save_dir / "thumbnails", save_dir / "audio"
    thumb_files = [p for n in doomed for p in thumbs_dir.glob(Path(n).stem + ".*")] if thumbs_dir.is_dir() else []
    audio_files = []
    if args.include_audio and audio_dir.is_dir():
        audio_files = [p for n in doomed for p in audio_dir.glob(Path(n).stem + ".*")]

    thumb_bytes = sum(p.stat().st_size for p in thumb_files)
    audio_bytes = sum(p.stat().st_size for p in audio_files)

    print(f"\nKeeping {n_kept:,}, removing {len(doomed):,} of {len(rows):,} frames "
          f"({100 * len(doomed) / len(rows):.0f}%)")
    print(f"  images     {human(image_bytes)}")
    print(f"  thumbnails {human(thumb_bytes)} ({len(thumb_files):,} files)")
    if args.include_audio:
        print(f"  audio      {human(audio_bytes)} ({len(audio_files):,} files)")
    else:
        print("  audio      untouched (pass --include-audio to remove it too)")
    print(f"  total      {human(image_bytes + thumb_bytes + audio_bytes)}")

    by_distance: dict[int, int] = defaultdict(int)
    for _name, _rep, dist in drops:
        by_distance[dist] += 1
    print("  by distance: " + "  ".join(f"{d}:{by_distance[d]:,}" for d in sorted(by_distance)))

    if not args.apply:
        print("\nDry run: nothing deleted. Re-run with --apply.")
        return 0

    removed = 0
    for name in doomed:
        (images_dir / name).unlink(missing_ok=True)
        removed += 1
    for p in thumb_files + audio_files:
        p.unlink(missing_ok=True)
    print(f"\nDeleted {removed:,} frames, {len(thumb_files):,} thumbnails, {len(audio_files):,} audio clips.")

    prune_metadata(save_dir, doomed)
    return 0


def prune_metadata(save_dir: Path, doomed: set[str]) -> None:
    """Drop records for removed frames so nothing dangles.

    Records key on bare filenames, so a removed frame leaves an entry that the
    review UI and the classifier would otherwise still resolve against.
    """
    labels_path = save_dir / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        kept = {k: v for k, v in labels.items() if k not in doomed}
        if len(kept) != len(labels):
            labels_path.write_text(json.dumps(kept, indent=1))
            print(f"  labels.json: dropped {len(labels) - len(kept):,} entries")

    for filename, key in (("features.jsonl", "filename"), ("classifications.jsonl", "filename")):
        path = save_dir / filename
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        kept_lines, dropped = [], 0
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            if record.get(key) in doomed:
                dropped += 1
            else:
                kept_lines.append(line)
        if dropped:
            path.write_text("\n".join(kept_lines) + "\n")
            print(f"  {filename}: dropped {dropped:,} records")


if __name__ == "__main__":
    sys.exit(main())
