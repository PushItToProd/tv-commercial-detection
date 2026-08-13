"""Build labelled contact sheets so frames can be ground-truthed in bulk.

The archive is 269 disjoint ~5-frame bursts, and within a burst the content
barely changes, so one representative per burst is enough for a first pass;
`--frames` renders an explicit list when a burst needs looking at in full.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

IMAGES = Path("/home/joe/Code/projects/tv-commercial-detector/server/frames/images")


def build(names: list[str], labels: list[str], out: Path, cols: int, tw: int, th: int) -> None:
    rows = (len(names) + cols - 1) // cols
    bar = 18
    sheet = Image.new("RGB", (cols * tw, rows * (th + bar)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for k, (name, lab) in enumerate(zip(names, labels)):
        r, c = divmod(k, cols)
        x, y = c * tw, r * (th + bar)
        try:
            with Image.open(IMAGES / name) as im:
                im = im.convert("RGB").resize((tw, th), Image.LANCZOS)
                sheet.paste(im, (x, y + bar))
        except Exception:
            draw.rectangle([x, y + bar, x + tw, y + bar + th], fill=(60, 0, 0))
        draw.rectangle([x, y, x + tw, y + bar], fill=(0, 0, 0))
        draw.text((x + 3, y + 3), lab, fill=(255, 255, 0))
        draw.rectangle([x, y, x + tw - 1, y + bar + th - 1], outline=(90, 90, 90))
    sheet.save(out, quality=88)
    print(f"{out}  {len(names)} frames  {sheet.size}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--mode", choices=["burst", "frames"], default="burst")
    ap.add_argument("--indices", default="")
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tw", type=int, default=320)
    ap.add_argument("--th", type=int, default=180)
    ap.add_argument("--sheets", default="")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.table)]
    by_i = {r["i"]: r for r in rows}

    if args.mode == "frames":
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
        picks = [by_i[i] for i in idxs]
        tags = [f"{r['i']} {r['timestamp'][11:19]} {r['live_class'][:4]}" for r in picks]
    else:
        picks, tags = [], []
        for b, burst in enumerate(json.load(open(args.table.replace(".jsonl", "_bursts.json")))):
            mid = burst[len(burst) // 2]
            r = by_i[mid]
            picks.append(r)
            tags.append(f"b{b} {r['timestamp'][11:19]} {r['live_class'][:4]}")

    want = {int(x) for x in args.sheets.split(",") if x.strip()}
    for s in range(0, len(picks), args.per_sheet):
        sheet_no = s // args.per_sheet
        if want and sheet_no not in want:
            continue
        build(
            [r["filename"] for r in picks[s : s + args.per_sheet]],
            tags[s : s + args.per_sheet],
            Path(f"{args.out_prefix}_{sheet_no:02d}.jpg"),
            args.cols, args.tw, args.th,
        )


if __name__ == "__main__":
    main()
