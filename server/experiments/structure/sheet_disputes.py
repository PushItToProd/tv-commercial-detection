"""One row per disputed run: hand-labelled vs furniture-proposed."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from ground_truth import build  # noqa: E402
from timeline import load  # noqa: E402

IMAGES = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/"
    "USA_4K_Iowa_Corn_350/images"
)
COLS = 8


def main():
    rows = load()
    segs = build(rows)
    hand = ["content"] * len(rows)
    for s in segs:
        for i in range(s["start"], s["end"] + 1):
            hand[i] = s["label"]
    prop = ["content"] * len(rows)
    for b in json.load(open(Path(__file__).parent / "proposed.json")):
        for i in range(b["start"], b["end"] + 1):
            prop[i] = "ad"

    dis = [i for i in range(len(rows)) if hand[i] != prop[i]]
    runs = []
    for i in dis:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    runs = [r for r in runs if r[1] - r[0] + 1 >= 3]

    tw, th, bar = 250, 141, 15
    per = 8
    for sh in range(0, len(runs), per):
        chunk = runs[sh : sh + per]
        img = Image.new("RGB", (COLS * tw, len(chunk) * (th + bar)), (20, 20, 20))
        d = ImageDraw.Draw(img)
        for r, (a, b) in enumerate(chunk):
            lo, hi = a - 2, b + 2
            idx = [lo + round(k * (hi - lo) / (COLS - 1)) for k in range(COLS)]
            for c, i in enumerate(idx):
                x, y = c * tw, r * (th + bar)
                if 0 <= i < len(rows):
                    with Image.open(IMAGES / rows[i]["filename"]) as im:
                        img.paste(
                            im.convert("RGB").resize((tw, th), Image.LANCZOS),
                            (x, y + bar),
                        )
                d.rectangle([x, y, x + tw, y + bar], fill=(0, 0, 0))
                inside = a <= i <= b
                d.text(
                    (x + 3, y + 2),
                    f"{'*' if inside else ' '}{i} h={hand[i][:3]} p={prop[i][:3]}",
                    fill=(255, 120, 120) if inside else (170, 170, 170),
                )
                d.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline=(90, 90, 90))
        out = f"{sys.argv[1]}/disp_{sh // per}.jpg"
        img.save(out, quality=85)
        print(out, [(a, b) for a, b in chunk])


if __name__ == "__main__":
    main()
