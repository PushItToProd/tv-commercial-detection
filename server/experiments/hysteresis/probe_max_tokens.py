"""Does capping max_tokens bound the latency tail without changing verdicts?

`classify_by_prompt` allows 500 tokens while a typical reply is under 30, so a
rare rambling generation is the whole reason the per-frame maximum reaches ~5 s.
This re-runs a sample weighted toward the slowest frames at 500 vs 100 tokens
and compares both the time and the answer.
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

HERE = Path(__file__).parent
SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
PROMPT = (SERVER / "src/tv_commercial_detector/prompt/prompt_nbc.txt").read_text()
URL = "http://gmktec.zane.network:3002/v1/chat/completions"
MODEL = "/models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"


def b64(path: Path) -> str:
    with Image.open(path) as img:
        img.thumbnail((800, 800), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
    return base64.b64encode(buf.getvalue()).decode()


def call(data: str, max_tokens: int):
    content = [{"type": "text", "text": PROMPT},
               {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}]
    t0 = time.perf_counter()
    r = requests.post(URL, json={"model": MODEL,
                                 "messages": [{"role": "user", "content": content}],
                                 "max_tokens": max_tokens, "temperature": 0.2}, timeout=300)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    d = r.json()
    reply = d["choices"][0]["message"]["content"]
    return dt, d["usage"]["completion_tokens"], reply


def verdict(reply: str) -> str:
    low = reply.strip().lower()
    if "type=ad" in low:
        return "ad"
    if "type=racing" in low:
        return "content"
    return "unknown"


def main() -> None:
    images = Path(sys.argv[1])
    replay = json.load(open(HERE / "cont_replay.json"))
    # Slowest frames first, plus a spread of ordinary ones for comparison.
    slow = sorted(
        ((max(r["secs"] for r in runs), fn) for fn, runs in replay.items()
         if runs[0]["reason"] == "model-match"), reverse=True)
    picked = [fn for _, fn in slow[:15]] + [fn for _, fn in slow[len(slow) // 2 :][:25]]

    rows = []
    for n, fn in enumerate(picked):
        data = b64(images / fn)
        for cap in (500, 100):
            dt, ctok, reply = call(data, cap)
            rows.append({"fn": fn, "cap": cap, "secs": dt, "tokens": ctok,
                         "verdict": verdict(reply)})
        if n % 10 == 0:
            print(f"  {n}/{len(picked)}", file=sys.stderr, flush=True)

    for cap in (500, 100):
        v = sorted(r["secs"] for r in rows if r["cap"] == cap)
        t = [r["tokens"] for r in rows if r["cap"] == cap]
        print(f"max_tokens={cap:3d}: median {st.median(v):.2f}s p90 {v[int(.9*len(v))]:.2f} "
              f"max {v[-1]:.2f}s | completion tokens median {st.median(t):.0f} max {max(t)}")

    by = {}
    for r in rows:
        by.setdefault(r["fn"], {})[r["cap"]] = r["verdict"]
    agree = sum(1 for d in by.values() if d.get(500) == d.get(100))
    print(f"verdict agreement between caps: {agree}/{len(by)}")
    json.dump(rows, open(HERE / "max_tokens_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
