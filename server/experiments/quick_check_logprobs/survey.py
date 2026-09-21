"""Ask both LLM passes, with logprobs, on a test set drawn from two broadcasts.

The test set is every frame the nascar_on_nbc cascade sends to the LLM in
production (a census: OpenCV undecided and the audio sensor unconfident or
abstaining), plus a sample of the frames OpenCV leaves undecided that the audio
sensor decides. The sensor's model will change and it abstains whenever capture
is dead or the interval is coarse, and then those frames reach the LLM too.

Where each frame falls in the cascade comes from `cascade_<broadcast>.jsonl`,
written by `scripts/fit_audio_model.py check --dump`. The audio model was fit on
Iowa, so its verdicts there are in-sample and send fewer frames to the LLM than
a held-out broadcast would.

Each sampled frame carries `weight`, the number of frames in the broadcast it
stands for, so rates over the audio-decided stratum can be scaled back up.

Four conditions per frame, one pass per condition so the prompt cache never
holds the previous condition's encoding of the same image:

    quick_image  quick-check image-only prompt, max_tokens=1
    quick_audio  quick-check audio prompt with the clip, max_tokens=1
    full_image   prompt_nbc.txt, image only
    full_audio   prompt_nbc.txt with the clip

Audio conditions are skipped for silent clips, which production withholds.
Results append to survey_raw.jsonl, one line per (condition, frame), and a
re-run skips what's already there.

    uv run python experiments/quick_check_logprobs/survey.py [--sample-only]
"""

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

from full_prompt import PROMPT, verdict_prob
from openai import OpenAI
from run import AUDIO_PROMPT, IMAGE_PROMPT, NO, YES

from tv_commercial_detector import audio_health
from tv_commercial_detector.classification import llm_match

HERE = Path(__file__).parent
BASE = Path("/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com")
BROADCASTS = ("USA_4K_Cook_Out_400", "USA_4K_Iowa_Corn_350")
MODEL = json.loads(
    (HERE.parent.parent / "src/tv_commercial_detector/classifiers/audio_models/nascar_on_nbc.json").read_text()
)
AUDIO_PER_VERDICT = 60  # audio-decided frames per verdict, across both broadcasts
CONDITIONS = ("quick_image", "quick_audio", "full_image", "full_audio")


def stage(row: dict) -> str:
    if row["anchor"] is not None:
        return "opencv"
    p = row["p_audio"]
    if p is not None and (p >= MODEL["ad_threshold"] or p <= MODEL["content_threshold"]):
        return "audio"
    return "llm"


def flag_group(ann: dict) -> str:
    """Stratum within a verdict: the operator's note, else a risk/care flag, else plain."""
    if ann.get("note"):
        return "note: " + ann["note"][:60]
    if ann.get("risk") == "fraught":
        return "fraught"
    if ann.get("care_away") and ann.get("care_back"):
        return "ambiguous"
    return "plain"


def build_sample(rng: random.Random) -> list[dict]:
    census, pools = [], defaultdict(list)
    for b in BROADCASTS:
        frames = json.loads((BASE / b / "annotations.json").read_text())["frames"]
        for line in (HERE / f"cascade_{b}.jsonl").open():
            row = json.loads(line)
            ann = frames.get(row["filename"], {})
            if ann.get("verdict") not in ("ad", "content") or ann.get("exclude"):
                continue
            s = stage(row)
            rec = {
                "broadcast": b,
                "filename": row["filename"],
                "stage": s,
                "anchor": row["anchor"],
                "p_audio": row["p_audio"],
                "verdict": ann["verdict"],
                "group": flag_group(ann),
                **{k: ann.get(k) for k in ("video", "audio", "risk", "care_away", "care_back", "note")},
            }
            if s == "llm":
                census.append({**rec, "weight": 1.0})
            elif s == "audio":
                pools[(ann["verdict"], b, rec["group"])].append(rec)

    # Audio-decided frames: half plain, half spread evenly over the flag groups,
    # split between broadcasts; weight is the stratum's size over its draw.
    picked = []
    for verdict in ("ad", "content"):
        per_b = AUDIO_PER_VERDICT // len(BROADCASTS)
        for b in BROADCASTS:
            groups = {g: v for (vv, bb, g), v in pools.items() if vv == verdict and bb == b}
            want = {"plain": per_b // 2} if "plain" in groups else {}
            flagged = sorted(g for g in groups if g != "plain")
            left = per_b - sum(want.values())
            while left > 0 and any(want.get(g, 0) < len(groups[g]) for g in flagged):
                for g in flagged:
                    if left and want.get(g, 0) < len(groups[g]):
                        want[g] = want.get(g, 0) + 1
                        left -= 1
            if left:  # not enough flagged frames; top up from plain
                want["plain"] = min(len(groups.get("plain", [])), want.get("plain", 0) + left)
            for g, k in want.items():
                for rec in rng.sample(groups[g], k):
                    picked.append({**rec, "weight": len(groups[g]) / k})
    return census + picked


def ask(client: OpenAI, content: list, quick: bool) -> dict:
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
        max_tokens=1 if quick else 500,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    wall_ms = (time.perf_counter() - t0) * 1000
    reply = r.choices[0].message.content or ""
    tokens = r.choices[0].logprobs.content  # pyright: ignore[reportOptionalMemberAccess]
    t = (r.model_extra or {}).get("timings", {})
    out = {
        "reply": reply,
        "wall_ms": wall_ms,
        **{k: t.get(k) for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")},
    }
    if quick:
        p_yes = p_no = 0.0
        for alt in tokens[0].top_logprobs:  # pyright: ignore[reportOptionalSubscript]
            norm = alt.token.strip().lower()
            if norm in YES:
                p_yes += math.exp(alt.logprob)
            elif norm in NO:
                p_no += math.exp(alt.logprob)
        out["p_yes"] = p_yes / (p_yes + p_no) if p_yes + p_no else None
    else:
        out["type"] = llm_match._get_classification_from_response(reply.strip().lower()).type
        out.update(verdict_prob(tokens))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://gmktec.zane.network:3002")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--sample-only", action="store_true", help="write survey_sample.jsonl and stop")
    args = ap.parse_args()

    sample_path = HERE / "survey_sample.jsonl"
    if sample_path.exists():
        sample = [json.loads(line) for line in sample_path.open()]
    else:
        sample = build_sample(random.Random(args.seed))
        with sample_path.open("w") as f:
            for rec in sample:
                f.write(json.dumps(rec) + "\n")
    by = defaultdict(int)
    for rec in sample:
        by[(rec["stage"], rec["verdict"])] += 1
    print(f"{len(sample)} frames: {dict(by)}")
    if args.sample_only:
        return

    raw_path = HERE / "survey_raw.jsonl"
    done = set()
    if raw_path.exists():
        for line in raw_path.open():
            r = json.loads(line)
            done.add((r["condition"], r["broadcast"], r["filename"]))

    client = OpenAI(base_url=f"{args.url}/v1", api_key="none")
    with raw_path.open("a") as out:
        for cond in CONDITIONS:
            t0 = time.perf_counter()
            for i, rec in enumerate(sample):
                key = (cond, rec["broadcast"], rec["filename"])
                if key in done:
                    continue
                root = BASE / rec["broadcast"]
                wav = root / "audio" / (Path(rec["filename"]).stem + ".wav")
                audio_bytes = wav.read_bytes() if wav.exists() else None
                silent = audio_bytes is None or audio_health.is_silent_clip(audio_bytes)
                if cond.endswith("audio") and silent:
                    continue
                b64 = llm_match.load_image_b64(str(root / "images" / rec["filename"]))
                img = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                quick = cond.startswith("quick")
                text = (AUDIO_PROMPT if cond == "quick_audio" else IMAGE_PROMPT) if quick else PROMPT
                content: list = [{"type": "text", "text": text}, img]
                if cond.endswith("audio"):
                    content.append(
                        {"type": "input_audio", "input_audio": {"data": llm_match.audio_b64(audio_bytes), "format": "wav"}}
                    )
                res = ask(client, content, quick)
                out.write(json.dumps({"condition": cond, "broadcast": rec["broadcast"],
                                      "filename": rec["filename"], **res}) + "\n")
                out.flush()
                if i % 50 == 0:
                    print(f"  {cond} {i}/{len(sample)} {time.perf_counter() - t0:.0f}s", flush=True)
            print(f"{cond} done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
