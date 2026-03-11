import json
import sys
from pathlib import Path

from classify import _classify_image, get_classification_from_response


# get all images in frames/ and classify them, then print out any that were
# classified incorrectly, including the full reply

def main():
    # read json from frames/labels.json to get the expected labels
    labels_path = Path("frames/labels.json")
    with labels_path.open() as f:
        labels = json.load(f)

    image_files = list((Path("frames")).glob("*.png"))
    num_incorrect = 0

    incorrectly_marked_as_ads = []
    incorrectly_marked_as_content = []
    incorrectly_unknown = []

    for f in image_files:
        actual = labels[f.name]
        resp = _classify_image(f.as_posix())
        classification = get_classification_from_response(resp)

        if classification == actual:
            print(f"{f.name}: correct ({classification})")
            continue
        print(f"{f.name}: incorrect (classified as {classification}, expected {actual})")

        num_incorrect += 1

        if classification == "ad":
            incorrectly_marked_as_ads.append(f.name)
        elif classification == "content":
            incorrectly_marked_as_content.append(f.name)
        else:
            incorrectly_unknown.append(f.name)

        print(f"  Model reply for incorrect classification: {resp}")

        sys.stdout.flush()

    print(f"Processed {len(image_files)} images, {num_incorrect} incorrect classifications.")
    print(f"Num. incorrectly marked as ads: {len(incorrectly_marked_as_ads)}")
    print(f"  Incorrectly marked as ads: {', '.join(incorrectly_marked_as_ads)}")
    print(f"Num. incorrectly marked as content: {len(incorrectly_marked_as_content)}")
    print(f"  Incorrectly marked as content: {', '.join(incorrectly_marked_as_content)}")
    print(f"Num. classified as unknown: {len(incorrectly_unknown)}")
    print(f"  Classified as unknown: {', '.join(incorrectly_unknown)}")


if __name__ == '__main__':
    main()
