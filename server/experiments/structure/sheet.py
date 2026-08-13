"""Contact sheets over a frame-index range of the continuous recording."""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from timeline import cv_verdict, load  # noqa: E402

IMAGES = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/"
    "USA_4K_Iowa_Corn_350/images"
)

TAG = {"ad": (255, 90, 90), "content": (120, 255, 120), None: (255, 255, 0)}


def build(rows, out, cols=8, tw=256, th=144):
    bar = 15
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, nrows * (th + bar)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        rr, c = divmod(k, cols)
        x, y = c * tw, rr * (th + bar)
        try:
            with Image.open(IMAGES / r["filename"]) as im:
                sheet.paste(
                    im.convert("RGB").resize((tw, th), Image.LANCZOS), (x, y + bar)
                )
        except Exception:
            draw.rectangle([x, y + bar, x + tw, y + bar + th], fill=(60, 0, 0))
        draw.rectangle([x, y, x + tw, y + bar], fill=(0, 0, 0))
        v = cv_verdict(r)
        draw.text(
            (x + 3, y + 2), f"{r['i']} {r['timestamp'][11:19]} {v or '-'}", fill=TAG[v]
        )
        draw.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline=(90, 90, 90))
    sheet.save(out, quality=85)
    print(f"{out}  {len(rows)} frames  {rows[0]['i']}-{rows[-1]['i']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", required=True, help="i0:i1 frame index range")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=8)
    args = ap.parse_args()

    rows = load()
    i0, i1 = (int(x) for x in args.range.split(":"))
    build(rows[i0 : i1 + 1 : args.step], args.out, cols=args.cols)


if __name__ == "__main__":
    main()
