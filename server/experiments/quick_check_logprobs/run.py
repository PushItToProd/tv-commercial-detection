"""How confident is the LLM quick check (`_report_racing_related`)?

Samples ~50 ad and ~50 content frames from an annotated recording, stratified so
the operator's noted hard cases are all represented, and asks the quick-check
question with logprobs on. Each frame is asked twice: the image-only prompt, and
the audio prompt with the frame's clip (production runs the audio prompt,
since config.json sets enable_llm_audio, whenever the clip isn't silent).

Writes one JSON line per frame to results.jsonl beside this script.

    uv run python experiments/quick_check_logprobs/run.py [--dir DIR] [--url URL]
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import cv2
from openai import OpenAI

from tv_commercial_detector import audio_health
from tv_commercial_detector.classification import llm_match
from tv_commercial_detector.classifiers import nascar_on_nbc

HERE = Path(__file__).parent
DEFAULT_DIR = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/tv.youtube.com/USA_4K_Cook_Out_400"
)
SUBJECT = "NASCAR racing"
IMAGE_PROMPT = (
    f"Does this image contain anything related to {SUBJECT}? Reply with only 'yes' or 'no'."
)
AUDIO_PROMPT = (
    "This image and audio clip are from the same segment of a video. "
    "Based on both the audio and the image, does it seem more likely than not "
    f"that this segment is from a {SUBJECT} broadcast (not an ad)? Reply 'Yes' or 'No'."
)

# (stratum name, predicate, how many to take). Taken in order; a frame goes to
# the first stratum that claims it, and "fill" tops each verdict up to 50.
AD_STRATA = [
    ("hamlin", lambda r: "Denny Hamlin" in r.get("note", ""), 5),
    ("briscoe", lambda r: "Chase Briscoe" in r.get("note", ""), 5),
    ("wallace_reddick", lambda r: "Bubba Wallace" in r.get("note", ""), 5),
    ("americana_promo", lambda r: "promo spot" in r.get("note", ""), 4),
    ("post_break_ad_read", lambda r: "post-ad-break" in r.get("note", ""), 5),
    ("ambiguous", lambda r: r.get("care_away") and r.get("care_back"), 3),
    ("fraught", lambda r: r.get("risk") == "fraught", 3),
    ("bumper", lambda r: r.get("video") == "bumper", 3),
    ("inset_crowd_live", lambda r: r.get("video") in ("inset", "crowd", "live_race"), 4),
    ("side_by_side", lambda r: r.get("video") == "side_by_side", 5),
    ("spot", lambda r: r.get("video") == "spot", 4),
]
CONTENT_STRATA = [
    ("diffey_segment", lambda r: "Leigh Diffey" in r.get("note", ""), 5),
    ("americana_ad_read", lambda r: "NASCAR Americana" in r.get("note", ""), 5),
    ("squeezeback", lambda r: "squeezeback" in r.get("note", ""), 4),
    ("ambiguous", lambda r: r.get("care_away") and r.get("care_back"), 5),
    ("fraught", lambda r: r.get("risk") == "fraught", 3),
    ("studio", lambda r: r.get("video") == "studio", 4),
    ("side_by_side", lambda r: r.get("video") == "side_by_side", 4),
    ("live_race", lambda r: r.get("video") == "live_race", 10),
]
PER_VERDICT = 50


def sample(frames: dict, rng: random.Random) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    taken: set[str] = set()
    for verdict, strata in (("ad", AD_STRATA), ("content", CONTENT_STRATA)):
        pool = sorted(
            f for f, r in frames.items() if r.get("verdict") == verdict and not r.get("exclude")
        )
        n = 0
        for name, pred, k in strata:
            cands = [f for f in pool if f not in taken and pred(frames[f])]
            for f in rng.sample(cands, min(k, len(cands))):
                chosen.append((f, name))
                taken.add(f)
                n += 1
        rest = [f for f in pool if f not in taken]
        for f in rng.sample(rest, PER_VERDICT - n):
            chosen.append((f, "fill"))
            taken.add(f)
    return chosen


def sample_undecided(frames: dict, root: Path, n: int, rng: random.Random) -> list[tuple[str, str]]:
    chosen = []
    for verdict in ("ad", "content"):
        pool = sorted(
            f for f, r in frames.items() if r.get("verdict") == verdict and not r.get("exclude")
        )
        rng.shuffle(pool)
        got = 0
        for f in pool:
            if got == n:
                break
            if opencv_verdict(root / "images" / f) is None:
                chosen.append((f, "undecided"))
                got += 1
    return chosen


def opencv_verdict(image_path: Path) -> str | None:
    """What the nascar_on_nbc OpenCV checks decide, or None if they leave it to later passes."""
    img = cv2.imread(str(image_path))
    if img.shape[:2] != (1080, 1920):
        img = cv2.resize(img, (1920, 1080))
    if nascar_on_nbc.has_side_by_side_logo(img):
        return "ad"
    if nascar_on_nbc.has_network_logo(img):
        return "content"
    return None


YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}


def ask(client: OpenAI, content: list) -> dict:
    r = client.chat.completions.create(
        model="local",
        messages=[{"role": "user", "content": content}],  # pyright: ignore[reportArgumentType]
        max_tokens=10,
        temperature=0,
        logprobs=True,
        top_logprobs=20,
    )
    choice = r.choices[0]
    first = choice.logprobs.content[0]  # pyright: ignore[reportOptionalMemberAccess, reportOptionalSubscript]
    p_yes = p_no = 0.0
    for t in first.top_logprobs:
        norm = t.token.strip().lower()
        if norm in YES:
            p_yes += math.exp(t.logprob)
        elif norm in NO:
            p_no += math.exp(t.logprob)
    return {
        "reply": choice.message.content,
        "p_yes": p_yes,
        "p_no": p_no,
        # P(yes) among yes/no answers; 1 - (p_yes + p_no) is mass on anything else.
        "p_yes_norm": p_yes / (p_yes + p_no) if p_yes + p_no else None,
        "top": [(t.token, round(t.logprob, 3)) for t in first.top_logprobs[:6]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--url", default="http://gmktec.zane.network:3002")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=Path, default=HERE / "results.jsonl")
    ap.add_argument(
        "--undecided",
        type=int,
        metavar="N",
        help="instead of the stratified sample, take N random ad and N random content"
        " frames that the OpenCV checks leave undecided (the frames production sends to the LLM)",
    )
    args = ap.parse_args()

    frames = json.loads((args.dir / "annotations.json").read_text())["frames"]
    rng = random.Random(args.seed)
    if args.undecided:
        chosen = sample_undecided(frames, args.dir, args.undecided, rng)
    else:
        chosen = sample(frames, rng)
    client = OpenAI(base_url=f"{args.url}/v1", api_key="none")

    counts: dict[str, int] = defaultdict(int)
    with args.out.open("w") as out:
        for i, (fname, stratum) in enumerate(chosen):
            ann = frames[fname]
            image_path = args.dir / "images" / fname
            wav = args.dir / "audio" / (Path(fname).stem + ".wav")
            audio_bytes = wav.read_bytes() if wav.exists() else None
            img_b64 = llm_match.load_image_b64(str(image_path))
            img_part = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}

            rec = {
                "filename": fname,
                "stratum": stratum,
                "verdict": ann.get("verdict"),
                "ambiguous": bool(ann.get("care_away") and ann.get("care_back")),
                **{k: ann.get(k) for k in ("video", "audio", "risk", "care_away", "care_back", "note")},
                "opencv": opencv_verdict(image_path),
                "clip_silent": audio_bytes is None or audio_health.is_silent_clip(audio_bytes),
                "image_only": ask(client, [{"type": "text", "text": IMAGE_PROMPT}, img_part]),
            }
            if not rec["clip_silent"]:
                audio_part = {
                    "type": "input_audio",
                    "input_audio": {"data": llm_match.audio_b64(audio_bytes), "format": "wav"},
                }
                rec["with_audio"] = ask(
                    client, [{"type": "text", "text": AUDIO_PROMPT}, img_part, audio_part]
                )
            out.write(json.dumps(rec) + "\n")
            out.flush()
            counts[rec["verdict"]] += 1
            print(
                f"{i + 1:3d} {rec['verdict']:7s} {stratum:18s} "
                f"img p_yes={rec['image_only']['p_yes_norm']:.3f} "
                + (f"aud p_yes={rec['with_audio']['p_yes_norm']:.3f}" if "with_audio" in rec else ""),
                flush=True,
            )


if __name__ == "__main__":
    main()
