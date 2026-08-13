"""Re-run the live nascar_on_nbc pipeline over the labelled window.

Runs the real `classify_image`, so the OpenCV short-circuits and both LLM passes
behave exactly as in production, and records per-frame verdict, reason, wall time
and the model's raw reply. Repeats give the per-frame verdict distribution, which
matters because the model is known to flip between identical runs at the
production temperature.

The reply is what makes a disagreement reviewable after the fact: a verdict alone
cannot distinguish the model misreading the frame from the frame being genuinely
ambiguous. `evaluate.py` ignores the field.

Audio is off: 61% of this day's clips are digital silence (see
notes/misclassification-analysis-2026-08.md), and feeding silence to the model
launders a false "live racing" signal into both LLM passes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

SERVER = Path("/home/joe/Code/projects/tv-commercial-detector/server")
sys.path.insert(0, str(SERVER / "src"))

from tv_commercial_detector.config import app_config  # noqa: E402

app_config.llm_url = "http://gmktec.zane.network:3002"
app_config.llm_model_name = (
    "/models/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M/Qwen3-Omni-30B-A3B-Instruct-Q4_K_M.gguf"
)
app_config.enable_llm_audio = False

from tv_commercial_detector.classifiers import nascar_on_nbc as nbc  # noqa: E402

IMAGES = SERVER / "frames" / "images"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dataset", default="dataset.jsonl")
    ap.add_argument("--images", default=str(IMAGES))
    args = ap.parse_args()

    images = Path(args.images)
    here = Path(__file__).parent
    rows = [json.loads(line) for line in open(here / args.dataset)]
    if args.limit:
        rows = rows[: args.limit]

    results: dict[str, list] = {}
    t_start = time.perf_counter()
    for rep in range(args.reps):
        for n, r in enumerate(rows):
            path = images / r["filename"]
            t0 = time.perf_counter()
            try:
                res = nbc.classify_image(str(path))
                verdict, reason, reply = res.type, res.reason, res.reply
            except Exception as e:
                verdict, reason, reply = "error", f"{type(e).__name__}", None
            dt = time.perf_counter() - t0
            results.setdefault(r["filename"], []).append(
                {"rep": rep, "type": verdict, "reason": reason, "secs": dt, "reply": reply}
            )
            if n % 200 == 0:
                el = time.perf_counter() - t_start
                print(f"rep {rep} {n}/{len(rows)}  {el:.0f}s", file=sys.stderr, flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"wrote {len(results)} frames x {args.reps} reps", file=sys.stderr)


if __name__ == "__main__":
    main()
