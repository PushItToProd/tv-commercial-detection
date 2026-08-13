"""One row per segment boundary: the frames either side of the snapped cut."""

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
BEFORE, AFTER = 3, 3


def main():
    rows = load()
    segs = build(rows)
    bounds = [
        (s["start"], segs[k - 1]["label"], s["label"], s["kind"])
        for k, s in enumerate(segs)
        if k > 0
    ]
    per = 8
    tw, th, bar = 250, 141, 15
    cols = BEFORE + AFTER
    for sh in range(0, len(bounds), per):
        chunk = bounds[sh : sh + per]
        img = Image.new("RGB", (cols * tw, len(chunk) * (th + bar)), (20, 20, 20))
        d = ImageDraw.Draw(img)
        for r, (b, prev, nxt, kind) in enumerate(chunk):
            for c, i in enumerate(range(b - BEFORE, b + AFTER)):
                x, y = c * tw, r * (th + bar)
                if 0 <= i < len(rows):
                    with Image.open(IMAGES / rows[i]["filename"]) as im:
                        img.paste(
                            im.convert("RGB").resize((tw, th), Image.LANCZOS),
                            (x, y + bar),
                        )
                d.rectangle([x, y, x + tw, y + bar], fill=(0, 0, 0))
                mark = "|>" if i == b else "  "
                d.text(
                    (x + 3, y + 2),
                    f"{mark}{i} {prev}->{nxt} {kind}",
                    fill=(255, 220, 0),
                )
                d.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline=(90, 90, 90))
        out = f"{sys.argv[1]}/bounds_{sh // per}.jpg"
        img.save(out, quality=85)
        print(out, [b for b, *_ in chunk])


if __name__ == "__main__":
    main()
