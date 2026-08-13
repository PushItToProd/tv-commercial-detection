"""Contact sheets for the continuous recording."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
IMAGES = Path("/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/"
              "USA_4K_Iowa_Corn_350/images")


def build(rows, out, cols=8, tw=240, th=135):
    bar = 16
    n = len(rows)
    nrows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, nrows * (th + bar)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows):
        rr, c = divmod(k, cols)
        x, y = c * tw, rr * (th + bar)
        try:
            with Image.open(IMAGES / r["filename"]) as im:
                sheet.paste(im.convert("RGB").resize((tw, th), Image.LANCZOS), (x, y + bar))
        except Exception:
            draw.rectangle([x, y + bar, x + tw, y + bar + th], fill=(60, 0, 0))
        draw.rectangle([x, y, x + tw, y + bar], fill=(0, 0, 0))
        draw.text((x + 3, y + 2), f"{r['i']} {r['timestamp'][11:19]}", fill=(255, 255, 0))
        draw.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline=(90, 90, 90))
    sheet.save(out, quality=88)
    print(f"{out}  {n} frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["anchors", "fallthrough"], required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--per-sheet", type=int, default=64)
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(HERE / "cont.jsonl")]
    if args.mode == "anchors":
        bug = [r for r in rows if r["usa"] >= 0.65 or r["peacock"] >= 0.55]
        sbs = [r for r in rows if r["sbs"] >= 0.8]
        picks = bug[:: max(1, len(bug) // 48)][:48] + sbs[:: max(1, len(sbs) // 16)][:16]
    else:
        picks = [r for r in rows
                 if not (r["usa"] >= 0.65 or r["peacock"] >= 0.55) and r["sbs"] < 0.8]
    for s in range(0, len(picks), args.per_sheet):
        build(picks[s : s + args.per_sheet], f"{args.prefix}_{s // args.per_sheet:02d}.jpg")


if __name__ == "__main__":
    main()
