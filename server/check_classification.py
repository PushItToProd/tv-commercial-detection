import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from classify import classify_image
import classify


IMAGES_DIR = Path(__file__).parent / "frames"
LABELS_PATH = IMAGES_DIR / "labels.json"


def get_images(base_dir: str | Path = IMAGES_DIR) -> list[Path]:
    base_dir = Path(base_dir)
    image_files = list(base_dir.glob("*.png")) + list(base_dir.glob("*.jpg"))
    image_files = sorted(image_files, key=lambda p: p.name)
    image_files = [f for f in image_files if not f.name.startswith("compressed_")]
    return image_files


# get all images in frames/ and classify them, then print out any that were
# classified incorrectly, including the full reply

def main():
    # read json from frames/labels.json to get the expected labels
    labels_path = LABELS_PATH
    images_dir = IMAGES_DIR

    with labels_path.open() as f:
        labels = json.load(f)

    image_files = get_images(images_dir)

    num_incorrect = 0
    num_unlabeled = 0
    num_ignored = 0

    incorrectly_marked_as_ads = []
    incorrectly_marked_as_content = []
    incorrectly_unknown = []
    times_taken = []

    for f in image_files:
        sys.stdout.flush()
        actual = labels.get(f.name, None)
        if actual is None:
            print(json.dumps({"file": f.name, "status": "unlabeled"}))
            num_unlabeled += 1
            continue
        elif actual == "ignore":
            print(json.dumps({"file": f.name, "status": "ignored"}))
            num_ignored += 1
            continue

        start_time = time.perf_counter()
        resp = classify_image(f.as_posix())
        elapsed_time = time.perf_counter() - start_time
        times_taken.append(elapsed_time)

        classification = resp.type
        reason = resp.reason
        model_reply = resp.reply

        if classification == actual:
            print(json.dumps({"file": f.name, "status": "correct", "expected": actual, "classified": classification, "elapsed": round(elapsed_time, 2), "model_reply": asdict(resp)}))
            continue

        num_incorrect += 1

        if classification == "ad":
            incorrectly_marked_as_ads.append(f.name)
        elif classification == "content":
            incorrectly_marked_as_content.append(f.name)
        else:
            incorrectly_unknown.append(f.name)

        print(json.dumps({"file": f.name, "status": "incorrect", "expected": actual, "classified": classification, "elapsed": round(elapsed_time, 2), "model_reply": asdict(resp)}))

    num_images = len(image_files)
    num_skipped = num_unlabeled + num_ignored
    actual_num_classified = num_images - num_skipped

    summary: dict = {
        "status": "summary",
        "total": actual_num_classified,
        "incorrect": num_incorrect,
        "incorrect_pct": round(num_incorrect / actual_num_classified, 4) if actual_num_classified else 0,
        "unlabeled": num_unlabeled,
        "ignored": num_ignored,
        "incorrectly_marked_as_ads": incorrectly_marked_as_ads,
        "incorrectly_marked_as_content": incorrectly_marked_as_content,
        "incorrectly_unknown": incorrectly_unknown,
    }
    if times_taken:
        sorted_times = sorted(times_taken)
        summary.update({
            "avg_elapsed": round(sum(times_taken) / len(times_taken), 2),
            "median_elapsed": round(sorted_times[len(sorted_times) // 2], 2),
            "min_elapsed": round(min(times_taken), 2),
            "max_elapsed": round(max(times_taken), 2),
            "total_elapsed": round(sum(times_taken), 2),
        })
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
