import os
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import imagehash
from PIL import Image

IMAGE_DIR = Path(__file__).parent.parent / "frames"
HASH_THRESHOLD = 10                   # hamming distance; lower = stricter


def file_hash(path):
    """Exact duplicate detection via MD5."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_hash_duplicates(image_paths, file_hash=file_hash):
    hash_groups: dict[str, list[Path]] = defaultdict(list)

    for p in image_paths:
        hash_groups[file_hash(p)].append(p)
    # => hash_groups is a dict with ms5 keys and lists of filenames as values

    # filter hash_groups to only entries with at least 2 members
    exact_dupes = {
        group_hash: filenames
        for group_hash, filenames
        in hash_groups.items()
        if len(filenames) > 1
    }
    return exact_dupes, hash_groups


def find_phash_duplicates(unique_paths: list[Path], threshold):
    image_phashes = {}
    for p in unique_paths:
        try:
            image_phashes[p] = imagehash.phash(Image.open(p))
        except Exception as e:
            print(json.dumps({"type": "error", "file": p.name, "message": str(e)}))

    paths_list = list(image_phashes.keys())
    near_dupe_groups = []
    visited = set()

    for i, p1 in enumerate(paths_list):
        if p1 in visited:
            continue
        group = [p1]
        for p2 in paths_list[i + 1:]:
            if p2 not in visited:
                if image_phashes[p1] - image_phashes[p2] <= threshold:
                    group.append(p2)
                    visited.add(p2)
        if len(group) > 1:
            near_dupe_groups.append(group)
            visited.add(p1)

    return near_dupe_groups, image_phashes


def find_duplicates(directory, threshold=HASH_THRESHOLD):
    image_paths = [
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and not p.name.startswith('compressed_')
    ]
    print(json.dumps({
        "type": "info",
        "message": f"Found {len(image_paths)} images"
    }))

    # --- Pass 1: exact duplicates by MD5 ---
    exact_dupes, hash_groups = find_hash_duplicates(image_paths)
    if exact_dupes:
        for hash, paths in exact_dupes.items():
            print(json.dumps({
                "type": "exact_duplicate",
                "hash": hash,
                "keep": paths[0].name,
                "dupes": [p.name for p in paths[1:]]
            }))

    # deduplicate returned paths
    unique_paths = [v[0] for v in hash_groups.values()]

    # --- Pass 2: near-duplicates via pHash ---
    near_dupe_groups, image_phashes = find_phash_duplicates(unique_paths, threshold)

    if near_dupe_groups:
        for group in near_dupe_groups:
            print(json.dumps({
                "type": "near_duplicate",
                "threshold": threshold,
                "group": [p.name for p in group],
                "hashes": {p.name: str(image_phashes[p]) for p in group}
            }))

    if not exact_dupes and not near_dupe_groups:
        print(json.dumps({
            "type": "info",
            "message": "No duplicates found."
        }))


def main():
    find_duplicates(IMAGE_DIR)


if __name__ == '__main__':
    main()
