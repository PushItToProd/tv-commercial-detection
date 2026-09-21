"""How confident is the full-prompt pass (`classify_by_prompt` with prompt_nbc.txt)?

Re-asks the frames already in results.jsonl and results_undecided.jsonl, so each
frame's full-prompt verdict can be set beside its quick-check scores. Verdicts
are re-read from annotations.json, picking up any corrections made since.

The reply is a short description ending in `type=ad` or `type=racing`, so the
verdict's probability is read at the token after the last "type=". That
probability is conditional on the description the model has already written.

Each frame is asked image-only and with its clip (production sends the clip
when enable_llm_audio is set and the clip isn't silent). Prompt caching is left
at llama.cpp's default, as in production: the prompt text precedes the image,
so its prefill is reused across requests and only the image and audio are
encoded fresh.

    uv run python experiments/quick_check_logprobs/full_prompt.py [--limit N]
"""

import argparse
import json
import math
import time
from pathlib import Path

from openai import OpenAI

from tv_commercial_detector.classification import llm_match

HERE = Path(__file__).parent
ROOT = Path("/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Cook_Out_400")
PROMPT = llm_match.load_prompt("prompt_nbc.txt")


def _split(alternatives, lead: str) -> tuple[float, float, float]:
    """Probability of alternatives continuing *lead* with ad, with racing, or with nothing yet."""
    p_ad = p_racing = p_bare = 0.0
    for t in alternatives:
        if not t.token.startswith(lead):
            continue
        rest = t.token[len(lead) :].strip().lower()
        p = math.exp(t.logprob)
        if not rest:
            p_bare += p
        elif "ad".startswith(rest) or rest.startswith("ad"):
            p_ad += p
        elif "racing".startswith(rest) or rest.startswith("racing"):
            p_racing += p
    return p_ad, p_racing, p_bare


def verdict_prob(tokens) -> dict:
    """P(ad) for the verdict after the last "type=" in the reply.

    Qwen's tokenizer writes "=r" as one token and "=" "ad" as two, so the choice
    is made at the token holding the "=": "=r…" is racing, and a bare "=" defers
    to the next token. When the reply took the bare "=", the next token's
    distribution splits that share; when it took "=r", the bare "=" is counted
    as ad, which is where it goes in every reply observed.
    """
    starts, text = [], ""
    for t in tokens:
        starts.append(len(text))
        text += t.token
    eq = text.lower().rfind("type=")
    if eq == -1:
        return {"p_ad": None, "top": None}
    eq += len("type")
    at = max(i for i, s in enumerate(starts) if s <= eq)
    lead = text[starts[at] : eq + 1]
    p_ad, p_racing, p_bare = _split(tokens[at].top_logprobs, lead)
    if tokens[at].token == lead and at + 1 < len(tokens):
        n_ad, n_racing, _ = _split(tokens[at + 1].top_logprobs, "")
        if n_ad + n_racing:
            p_ad += p_bare * n_ad / (n_ad + n_racing)
            p_racing += p_bare * n_racing / (n_ad + n_racing)
    else:
        p_ad += p_bare
    return {
        "p_ad": p_ad / (p_ad + p_racing) if p_ad + p_racing else None,
        "off": 1 - p_ad - p_racing,
        "top": [(t.token, round(t.logprob, 3)) for t in tokens[at].top_logprobs[:6]],
    }


def ask(client: OpenAI, content: list) -> dict:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
        max_tokens=500,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    wall = time.perf_counter() - t0
    reply = r.choices[0].message.content or ""
    timings = (r.model_extra or {}).get("timings", {})
    return {
        "reply": reply,
        "type": llm_match._get_classification_from_response(reply.strip().lower()).type,
        **verdict_prob(r.choices[0].logprobs.content),  # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]
        "wall_ms": wall * 1000,
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "cache_n": timings.get("cache_n"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://gmktec.zane.network:3002")
    ap.add_argument("--limit", type=int, help="stop after N frames (for timing)")
    ap.add_argument("--out", type=Path, default=HERE / "results_full_prompt.jsonl")
    args = ap.parse_args()
    client = OpenAI(base_url=f"{args.url}/v1", api_key="none")
    frames = json.loads((ROOT / "annotations.json").read_text())["frames"]

    prior = []
    for name in ("results.jsonl", "results_undecided.jsonl"):
        prior += [(name, json.loads(line)) for line in (HERE / name).open()]
    if args.limit:
        prior = prior[: args.limit]

    # One pass per condition: asking both conditions back to back would let the
    # second reuse the first's encoded image from the prompt cache.
    parts = {}
    for fname in (q["filename"] for _, q in prior):
        b64 = llm_match.load_image_b64(str(ROOT / "images" / fname))
        wav = (ROOT / "audio" / (Path(fname).stem + ".wav")).read_bytes()
        parts[fname] = (
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "input_audio", "input_audio": {"data": llm_match.audio_b64(wav), "format": "wav"}},
        )
    answers: dict[str, dict] = {"with_audio": {}, "image_only": {}}
    for mode, answered in answers.items():
        for i, (_, q) in enumerate(prior):
            img, aud = parts[q["filename"]]
            content = [{"type": "text", "text": PROMPT}, img] + ([aud] if mode == "with_audio" else [])
            a = answered[q["filename"]] = ask(client, content)
            p = "None" if a["p_ad"] is None else f"{a['p_ad']:.3f}"
            print(
                f"{mode:10s} {i + 1:3d} {frames[q['filename']].get('verdict'):7s} {q['stratum']:18s}"
                f" {a['type']:7s} p_ad={p} {a['wall_ms']:5.0f}ms {a['predicted_n']}tok",
                flush=True,
            )

    with args.out.open("w") as out:
        for sample, q in prior:
            fname = q["filename"]
            rec = {
                "filename": fname,
                "sample": sample,
                "stratum": q["stratum"],
                "verdict": frames[fname].get("verdict"),
                "verdict_before": q["verdict"],
                "ambiguous": q["ambiguous"],
                "risk": q["risk"],
                "note": q["note"],
                "opencv": q["opencv"],
                "quick_image_p_yes": q["image_only"]["p_yes_norm"],
                "quick_audio_p_yes": q["with_audio"]["p_yes_norm"],
                **{mode: answered[fname] for mode, answered in answers.items()},
            }
            out.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
