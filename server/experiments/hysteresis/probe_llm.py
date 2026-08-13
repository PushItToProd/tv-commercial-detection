"""Probe llama.cpp latency and prompt-token cost for 1..N image prompts."""
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
MAX_DIMENSION = 800


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
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=180,
    )
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    return dt, d["usage"], d["choices"][0]["message"]["content"]


def main():
    frames = sorted(p.name for p in IMAGES.glob("2026-08-09T21-1*.jpg"))[:8]
    paths = [IMAGES / f for f in frames]
    imgs = [b64(p) for p in paths]
    prompt = (PROMPT_DIR / "prompt_nbc.txt").read_text()
    print(f"prompt chars={len(prompt)}")

    def img_part(d):
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{d}"}}

    trials = {
        "quick_check_1img": (
            [{"type": "text", "text": "Does this image contain anything related to NASCAR"
              " racing? Reply with only 'yes' or 'no'."}, img_part(imgs[0])], 10),
        "full_prompt_1img": ([{"type": "text", "text": prompt}, img_part(imgs[0])], 500),
        "full_prompt_2img": (
            [{"type": "text", "text": prompt}] + [img_part(i) for i in imgs[:2]], 500),
        "full_prompt_3img": (
            [{"type": "text", "text": prompt}] + [img_part(i) for i in imgs[:3]], 500),
        "full_prompt_5img": (
            [{"type": "text", "text": prompt}] + [img_part(i) for i in imgs[:5]], 500),
        "full_prompt_8img": (
            [{"type": "text", "text": prompt}] + [img_part(i) for i in imgs[:8]], 500),
    }

    results = {}
    for name, (content, mt) in trials.items():
        lat, usage, reply = [], None, None
        for rep in range(3):
            try:
                dt, u, c = call(content, mt)
            except Exception as e:
                print(f"{name}: FAILED {type(e).__name__}: {str(e)[:300]}")
                lat = None
                break
            lat.append(dt)
            usage, reply = u, c
        if lat:
            results[name] = {"lat": lat, "usage": usage}
            print(f"{name:20s} lat {min(lat):.2f}/{st.median(lat):.2f}/{max(lat):.2f}s "
                  f" prompt_tok={usage['prompt_tokens']} completion={usage['completion_tokens']}"
                  f"  reply={reply[:80]!r}")
    Path(sys.argv[1]).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
