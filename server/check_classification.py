import json
import sys
import time
from pathlib import Path

from classify import _classify_image, get_classification_from_response
import classify


# get all images in frames/ and classify them, then print out any that were
# classified incorrectly, including the full reply

def main():
    # read json from frames/labels.json to get the expected labels
    labels_path = Path("frames/labels.json")
    with labels_path.open() as f:
        labels = json.load(f)

    # classify.EXAMPLES = classify.load_examples()
    classify.EXAMPLES = []

    image_files = list((Path("frames")).glob("*.png")) + list((Path("frames")).glob("*.jpg"))
    image_files = sorted(image_files, key=lambda p: p.name)
    image_files = [f for f in image_files if not f.name.startswith("compressed_")]
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
            print(f"{f.name}: no label found, skipping")
            num_unlabeled += 1
            continue
        elif actual == "ignore":
            print(f"{f.name}: label is 'ignore', skipping")
            num_ignored += 1
            continue

        start_time = time.perf_counter()
        resp = _classify_image(f.as_posix())
        elapsed_time = time.perf_counter() - start_time
        times_taken.append(elapsed_time)

        classification = get_classification_from_response(resp)

        if classification == actual:
            print(f"{f.name}: correct ({classification}) - {elapsed_time:.2f}s")
            continue

        print(f"{f.name}: incorrect (classified as {classification}, expected {actual}) - {elapsed_time:.2f}s")

        num_incorrect += 1

        if classification == "ad":
            incorrectly_marked_as_ads.append(f.name)
        elif classification == "content":
            incorrectly_marked_as_content.append(f.name)
        else:
            incorrectly_unknown.append(f.name)

        print(f"  Model reply for incorrect classification: {resp}")

    num_images = len(image_files)
    num_skipped = num_unlabeled + num_ignored
    actual_num_classified = num_images - num_skipped
    print(f"Processed {actual_num_classified} images, {num_incorrect} incorrect classifications ({num_incorrect/actual_num_classified:.2%}).")
    print(f"{num_unlabeled} unlabeled, {num_ignored} marked 'ignore'.")
    print()
    print(f"Num. incorrectly marked as ads: {len(incorrectly_marked_as_ads)} ({len(incorrectly_marked_as_ads)/actual_num_classified:.2%})")
    print(f"  Incorrectly marked as ads: {', '.join(incorrectly_marked_as_ads)}")
    print(f"Num. incorrectly marked as content: {len(incorrectly_marked_as_content)} ({len(incorrectly_marked_as_content)/actual_num_classified:.2%})")
    print(f"  Incorrectly marked as content: {', '.join(incorrectly_marked_as_content)}")
    print(f"Num. classified as unknown: {len(incorrectly_unknown)} ({len(incorrectly_unknown)/actual_num_classified:.2%})")
    print(f"  Classified as unknown: {', '.join(incorrectly_unknown)}")
    print()
    print(f"Average classification time: {sum(times_taken)/len(times_taken):.2f}s")
    print(f"Median classification time: {sorted(times_taken)[len(times_taken)//2]:.2f}s")
    print(f"Min classification time: {min(times_taken):.2f}s")
    print(f"Max classification time: {max(times_taken):.2f}s")
    total_time_secs = sum(times_taken)
    print(f"Total classification time: {total_time_secs:.2f}s")


if __name__ == '__main__':
    main()
