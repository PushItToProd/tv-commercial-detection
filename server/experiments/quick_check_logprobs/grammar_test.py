"""Does a yes/no grammar change the quick check's speed or probabilities?

Re-asks the frames in results_undecided.jsonl under three conditions, rotating
the order per frame so no condition always runs first:

    baseline   max_tokens=10, as production
    grammar    max_tokens=10 plus a GBNF grammar allowing only yes/no in any case
    one_token  max_tokens=1, no grammar

Prompt caching is off so every request pays the full image prefill. Timings come
from llama.cpp's own `timings` block, not wall clock alone.

    uv run python experiments/quick_check_logprobs/grammar_test.py [--audio]
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from openai import OpenAI

from tv_commercial_detector.classification import llm_match

HERE = Path(__file__).parent
ROOT = Path("/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Cook_Out_400")
IMAGE_PROMPT = (
    "Does this image contain anything related to NASCAR racing? Reply with only 'yes' or 'no'."
)
AUDIO_PROMPT = (
    "This image and audio clip are from the same segment of a video. "
    "Based on both the audio and the image, does it seem more likely than not "
    "that this segment is from a NASCAR racing broadcast (not an ad)? Reply 'Yes' or 'No'."
)
GRAMMAR = 'root ::= [yY] [eE] [sS] | [nN] [oO]'
CONDITIONS = {
    "baseline": {"max_tokens": 10, "extra_body": {"cache_prompt": False}},
    "grammar": {"max_tokens": 10, "extra_body": {"cache_prompt": False, "grammar": GRAMMAR}},
    "one_token": {"max_tokens": 1, "extra_body": {"cache_prompt": False}},
}


def p_yes(first) -> tuple[float, float]:
    y = n = 0.0
    for t in first.top_logprobs:
        norm = t.token.strip().lower()
        if norm in ("yes", "y"):
            y += math.exp(t.logprob)
        elif norm in ("no", "n"):
            n += math.exp(t.logprob)
    return y / (y + n), 1 - y - n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", action="store_true", help="use the audio prompt with the clip")
    ap.add_argument("--url", default="http://gmktec.zane.network:3002")
    args = ap.parse_args()
    client = OpenAI(base_url=f"{args.url}/v1", api_key="none")

    rows = [json.loads(line) for line in (HERE / "results_undecided.jsonl").open()]
    out: dict[str, dict[str, list]] = {c: {"wall": [], "prompt_ms": [], "pred_ms": [],
                                           "pred_n": [], "p": [], "off": [], "reply": []}
                                       for c in CONDITIONS}
    names = list(CONDITIONS)
    for i, r in enumerate(rows):
        b64 = llm_match.load_image_b64(str(ROOT / "images" / r["filename"]))
        content: list = [
            {"type": "text", "text": AUDIO_PROMPT if args.audio else IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        if args.audio:
            wav = (ROOT / "audio" / (Path(r["filename"]).stem + ".wav")).read_bytes()
            content.append({"type": "input_audio",
                            "input_audio": {"data": llm_match.audio_b64(wav), "format": "wav"}})
        for c in names[i % 3:] + names[: i % 3]:
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model="local",
                messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                **CONDITIONS[c],
            )
            wall = time.perf_counter() - t0
            timings = resp.model_extra.get("timings", {})  # pyright: ignore[reportOptionalMemberAccess]
            p, off = p_yes(resp.choices[0].logprobs.content[0])  # pyright: ignore[reportOptionalMemberAccess, reportOptionalSubscript]
            o = out[c]
            o["wall"].append(wall)
            o["prompt_ms"].append(timings.get("prompt_ms"))
            o["pred_ms"].append(timings.get("predicted_ms"))
            o["pred_n"].append(timings.get("predicted_n"))
            o["p"].append(p)
            o["off"].append(off)
            o["reply"].append(resp.choices[0].message.content)

    print(f"{'audio' if args.audio else 'image-only'} prompt, n={len(rows)}")
    for c, o in out.items():
        print(
            f"  {c:10s} wall median {statistics.median(o['wall']) * 1000:6.0f} ms"
            f"  prefill median {statistics.median(o['prompt_ms']):6.0f} ms"
            f"  generation median {statistics.median(o['pred_ms']):5.1f} ms"
            f" over {statistics.mean(o['pred_n']):.1f} tokens"
            f"  max mass off yes/no {max(o['off']):.4f}"
        )
    base = out["baseline"]
    for c in ("grammar", "one_token"):
        d = [abs(a - b) for a, b in zip(base["p"], out[c]["p"])]
        flips = sum((a > 0.5) != (b > 0.5) for a, b in zip(base["p"], out[c]["p"]))
        print(f"  {c} vs baseline: max |dP(yes)| {max(d):.4f}, verdict flips {flips}")
    print("  distinct replies:", {c: sorted(set(o["reply"])) for c, o in out.items()})


if __name__ == "__main__":
    main()
