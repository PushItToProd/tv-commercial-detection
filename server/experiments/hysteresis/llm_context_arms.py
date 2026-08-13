"""Context-augmented LLM arms: does the model classify better with history?

The temporal policies in `arms.py` smooth the classifier's output. These change
its input instead, giving the model what the per-frame call cannot see:

  base      the production call - one image, prompt_nbc.txt
  images    the two preceding frames as extra images, chronological
  text      a CONTEXT block naming the recent verdicts and how long since the
            upper-right network bug was last matched
  both      images + text

Only frames the OpenCV pass does not settle are worth testing - the rest never
reach the model. Arms are interleaved per frame so that drift in server state or
load hits every arm equally, which the analysis established is necessary: single
frames flip verdict between otherwise identical runs at temperature 0.2.
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).parent
SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
IMAGES = SERVER / "frames" / "images"
PROMPT = (SERVER / "src/tv_commercial_detector/prompt/prompt_nbc.txt").read_text()

URL = "http://gmktec.zane.network:3002/v1/chat/completions"
MODEL = "/models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
MAX_DIMENSION = 800

MULTI_IMAGE_PREAMBLE = (
    "You are given several screenshots captured about two seconds apart from the"
    " same broadcast, in chronological order. Classify only the LAST image. The"
    " earlier images show what was on screen in the seconds just before it, and"
    " are there to help you tell a hard shot within an ongoing segment from a"
    " real cut between racing and a commercial break.\n\n"
)

CONTEXT_TEMPLATE = """
CONTEXT FROM THE PRECEDING SECONDS:
- Verdicts on the previous frames, oldest first: {history}
- Upper-right network bug (USA wordmark or NBC peacock) last matched: {bug}
- "NASCAR NON STOP" side-by-side banner last matched: {sbs}

Treat this as a prior, not an answer. Breaks and racing both run for minutes at a
time, so a single frame that disagrees with a run of recent frames is more often
a hard shot inside the same segment than a real transition. A long stretch with
no network bug at all is itself evidence of a break. Where this frame's own
evidence is clear, follow the evidence.
"""

_b64_cache: dict[str, str] = {}


def b64(name: str) -> str:
    if name not in _b64_cache:
        with Image.open(IMAGES / name) as img:
            img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=50)
        _b64_cache[name] = base64.b64encode(buf.getvalue()).decode()
    return _b64_cache[name]


def img_part(name: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64(name)}"}}


def call(content: list, max_tokens: int = 500) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens, "temperature": 0.2}, timeout=300)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], dt


def parse(reply: str) -> str:
    low = reply.strip().lower()
    if "type=ad" in low or '"classification": "ad"' in low:
        return "ad"
    if "type=racing" in low or '"classification": "racing"' in low:
        return "content"
    return "unknown"


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "not in the last few minutes"
    return f"{seconds:.0f} s ago"


def build(arm: str, row: dict, prior_names: list[str], history: list[str],
          bug_ago: float | None, sbs_ago: float | None) -> list:
    text = PROMPT
    parts: list = []
    if arm in ("images", "both") and prior_names:
        text = MULTI_IMAGE_PREAMBLE + text
    if arm in ("text", "both"):
        text = text + "\n" + CONTEXT_TEMPLATE.format(
            history=", ".join(history) if history else "none available",
            bug=ago(bug_ago), sbs=ago(sbs_ago))
    parts.append({"type": "text", "text": text})
    if arm in ("images", "both"):
        for n in prior_names:
            parts.append(img_part(n))
    parts.append(img_part(row["filename"]))
    return parts


def main() -> None:
    global IMAGES
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="base,images,text,both")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--prior", type=int, default=2, help="how many earlier frames to use")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--replay", default="replay.json")
    ap.add_argument("--images", default=str(IMAGES))
    args = ap.parse_args()
    IMAGES = Path(args.images)

    rows = [json.loads(line) for line in open(HERE / args.dataset)]
    replay = json.load(open(HERE / args.replay))

    by_index = {r["i"]: r for r in rows}
    order = [r["i"] for r in rows]
    pos = {i: k for k, i in enumerate(order)}

    def verdict_of(i: int) -> str | None:
        r = by_index[i]
        runs = replay.get(r["filename"])
        return runs[0]["type"] if runs else None

    # Only frames the OpenCV pass leaves undecided reach the model at all.
    targets = []
    for r in rows:
        runs = replay.get(r["filename"])
        if not runs:
            continue
        if runs[0]["reason"] in ("side_by_side", "network_logo", "phash_override"):
            continue
        if r["gt"] == "uncertain":
            continue
        targets.append(r)
    # Shuffled so a partial run is still a representative sample rather than
    # whichever commercial break happens to come first.
    import random
    random.Random(20260812).shuffle(targets)
    if args.limit:
        targets = targets[: args.limit]
    print(f"{len(targets)} target frames", file=sys.stderr)

    arms = args.arms.split(",")
    out: list[dict] = []
    t_start = time.perf_counter()

    for n, r in enumerate(targets):
        k = pos[r["i"]]
        # Earlier frames from the same episode only; a 50 s gap is not context.
        priors = []
        for back in range(args.prior, 0, -1):
            if k - back >= 0 and rows[k - back]["episode"] == r["episode"]:
                priors.append(rows[k - back])
        prior_names = [p["filename"] for p in priors]
        history = []
        for p in priors:
            v = verdict_of(p["i"])
            history.append({"ad": "ad", "content": "racing"}.get(v, "unclear"))

        bug_ago = sbs_ago = None
        import datetime as dt
        now = dt.datetime.fromisoformat(r["timestamp"])
        for back in range(1, 40):
            j = k - back
            if j < 0 or rows[j]["episode"] != r["episode"]:
                break
            q = rows[j]
            qt = dt.datetime.fromisoformat(q["timestamp"])
            if bug_ago is None and (q["usa"] >= 0.65 or q["peacock"] >= 0.55):
                bug_ago = (now - qt).total_seconds()
            if sbs_ago is None and q["sbs"] >= 0.8:
                sbs_ago = (now - qt).total_seconds()
        if r["usa"] >= 0.65 or r["peacock"] >= 0.55:
            bug_ago = 0.0

        for rep in range(args.reps):
            for arm in arms:
                content = build(arm, r, prior_names, history, bug_ago, sbs_ago)
                try:
                    reply, secs = call(content)
                    verdict = parse(reply)
                except Exception as e:
                    reply, secs, verdict = f"ERROR {type(e).__name__}: {e}", 0.0, "error"
                out.append({"i": r["i"], "filename": r["filename"], "gt": r["gt"],
                            "arm": arm, "rep": rep, "verdict": verdict,
                            "secs": secs, "reply": reply[:300]})
        if n % 25 == 0:
            el = time.perf_counter() - t_start
            print(f"{n}/{len(targets)}  {el:.0f}s", file=sys.stderr, flush=True)
            Path(args.out).write_text(json.dumps(out))

    Path(args.out).write_text(json.dumps(out))
    print(f"wrote {len(out)} verdicts", file=sys.stderr)


if __name__ == "__main__":
    main()
