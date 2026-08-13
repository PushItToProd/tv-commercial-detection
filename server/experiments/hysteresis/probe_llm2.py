"""Probe llama.cpp latency/token cost with a DISTINCT image per repetition.

The first probe reused one image across reps, so llama.cpp's prompt cache served
every call after the first and the timings were meaningless. Here each rep uses
a fresh frame, which is what production sees: the text prompt prefix stays
cached, the image never is.
"""
import base64
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import requests
from PIL import Image

URL = "http://gmktec.zane.network:3002/v1/chat/completions"
MODEL = "/models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
IMAGES = Path("/home/joe/Code/projects/tv-commercial-detector/server/frames/images")
PROMPT_DIR = Path(
    "/home/joe/Code/projects/tv-commercial-detector/server/src/tv_commercial_detector/prompt"
)
AUDIO = Path("/home/joe/Code/projects/tv-commercial-detector/server/frames/audio")
MAX_DIMENSION = 800
REPS = 8


def b64(path: Path) -> str:
    with Image.open(path) as img:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode()


def call(content, max_tokens):
    t0 = time.perf_counter()
    r = requests.post(
        URL,
        json={"model": MODEL, "messages": [{"role": "user", "content": content}],
              "max_tokens": max_tokens, "temperature": 0.2},
        timeout=300,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    return dt, d["usage"], d["choices"][0]["message"]["content"]


def img_part(d):
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}}


def main():
    # A long stretch of distinct frames so no two reps share an image.
    names = sorted(p.name for p in IMAGES.glob("2026-08-09T2*.jpg"))
    step = max(1, len(names) // (REPS * 12))
    picked = names[::step][: REPS * 10]
    cache = {}

    def get(n):
        if n not in cache:
            cache[n] = b64(IMAGES / n)
        return cache[n]

    prompt = (PROMPT_DIR / "prompt_nbc.txt").read_text()
    quick = ("Does this image contain anything related to NASCAR racing?"
             " Reply with only 'yes' or 'no'.")

    configs = {
        "quick_check_1img": ("quick", 1, 10),
        "full_prompt_1img": ("full", 1, 500),
        "full_prompt_2img": ("full", 2, 500),
        "full_prompt_3img": ("full", 3, 500),
        "full_prompt_5img": ("full", 5, 500),
    }

    results = {}
    cursor = 0
    for name, (kind, nimg, mt) in configs.items():
        lats, ptoks, ctoks, replies = [], [], [], []
        for rep in range(REPS):
            frames = [picked[(cursor + k) % len(picked)] for k in range(nimg)]
            cursor += nimg
            text = quick if kind == "quick" else prompt
            content = [{"type": "text", "text": text}] + [img_part(get(f)) for f in frames]
            try:
                dt, u, c = call(content, mt)
            except Exception as e:
                print(f"{name}: FAILED {type(e).__name__}: {str(e)[:200]}")
                lats = []
                break
            lats.append(dt)
            ptoks.append(u["prompt_tokens"])
            ctoks.append(u["completion_tokens"])
            replies.append(c)
        if not lats:
            continue
        results[name] = {"lat": lats, "prompt_tokens": ptoks, "completion_tokens": ctoks,
                         "replies": replies}
        s = sorted(lats)
        print(f"{name:20s} lat min {s[0]:.2f} med {st.median(s):.2f} p95 {s[int(.95*len(s))]:.2f}"
              f" max {s[-1]:.2f}s | prompt_tok {min(ptoks)}-{max(ptoks)}"
              f" | completion {min(ctoks)}-{max(ctoks)}")

    Path(sys.argv[1]).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
