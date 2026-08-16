"""Benchmark Qwen3-Omni against the recorded NBC/USA broadcast.

Each image/WAV pair is sent to every configured llama.cpp server with both the
audio-aware quick-reject prompt and ``prompt_nbc.txt``.  llama.cpp's response
timings measure prompt processing and token generation; a local wall clock
around the entire HTTP request measures end-to-end latency.

Examples:

    uv run python scripts/benchmark_qwen3_omni.py --limit 10
    uv run python scripts/benchmark_qwen3_omni.py --output benchmark.jsonl
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from tv_commercial_detector.classification.llm_match import load_image_b64, load_prompt

DEFAULT_BROADCAST_DIR = Path(
    "/mnt/data/tv-commercial-detector/full_broadcasts/"
    "tv.youtube.com/USA_4K_Iowa_Corn_350"
)
# ROUTER_MODEL = "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M"
ROUTER_MODEL = "Qwen3-Omni-30B-A3B-Instruct-Q4_K_M:CLASSIFIER"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

AUDIO_QUICK_REJECT_PROMPT = (
    "This image and audio clip are from the same segment of a video. "
    "Based on both the audio and the image, does it seem more likely than not "
    "that this segment is from a NASCAR race broadcast (not an ad)? Reply 'Yes' or 'No'."
)


@dataclass(frozen=True)
class Server:
    name: str
    url: str
    model: str


@dataclass(frozen=True)
class PromptCase:
    name: str
    text: str
    max_tokens: int


@dataclass(frozen=True)
class MediaPair:
    image: Path
    audio: Path


@dataclass
class Sample:
    host: str
    prompt: str
    filename: str
    e2e_ms: float
    prompt_ms: float | None = None
    generation_ms: float | None = None
    prompt_tokens: int | None = None
    cached_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    response: str | None = None
    error: str | None = None

    @property
    def overhead_ms(self) -> float | None:
        """Time not attributed by llama.cpp to prompt eval or generation."""
        if self.prompt_ms is None or self.generation_ms is None:
            return None
        return self.e2e_ms - self.prompt_ms - self.generation_ms


SERVERS = (
    Server("gmktec", "http://gmktec.zane.network:3002", "local"),
    Server("ai2", "http://ai2.zane.network:3000", ROUTER_MODEL),
    Server("framework", "http://framework.zane.network:3000", ROUTER_MODEL),
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3-Omni image+audio classification on three llama.cpp servers."
    )
    parser.add_argument(
        "--broadcast-dir",
        type=Path,
        default=DEFAULT_BROADCAST_DIR,
        help=f"recording root (default: {DEFAULT_BROADCAST_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="process only the first N image+audio pairs",
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=600.0,
        help="per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write one detailed JSON record per request (must not already exist)",
    )
    return parser.parse_args(argv)


def discover_pairs(root: Path, limit: int | None = None) -> tuple[list[MediaPair], int]:
    images_dir = root / "images"
    audio_dir = root / "audio"
    if not images_dir.is_dir():
        raise ValueError(f"image directory does not exist: {images_dir}")
    if not audio_dir.is_dir():
        raise ValueError(f"audio directory does not exist: {audio_dir}")

    pairs: list[MediaPair] = []
    missing_audio = 0
    images = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for image in images:
        audio = audio_dir / f"{image.stem}.wav"
        if not audio.is_file():
            missing_audio += 1
            continue
        pairs.append(MediaPair(image=image, audio=audio))
        if limit is not None and len(pairs) >= limit:
            break
    return pairs, missing_audio


def build_payload(
    server: Server, prompt: PromptCase, image_b64: str, audio_b64: str
) -> dict[str, Any]:
    return {
        "model": server.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            }
        ],
        "max_tokens": prompt.max_tokens,
        "temperature": 0.2,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _response_text(data: dict[str, Any]) -> str | None:
    try:
        content = data["choices"][0]["message"]["content"]
    except KeyError, IndexError, TypeError:
        return None
    return content if isinstance(content, str) else None


def run_request(
    client: httpx.Client,
    server: Server,
    prompt: PromptCase,
    pair: MediaPair,
    image_b64: str,
    audio_b64: str,
) -> Sample:
    payload = build_payload(server, prompt, image_b64, audio_b64)
    started = time.perf_counter()
    try:
        response = client.post(
            f"{server.url.rstrip('/')}/v1/chat/completions", json=payload
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("response JSON is not an object")
        timings = data.get("timings", {})
        if not isinstance(timings, dict):
            timings = {}
        return Sample(
            host=server.name,
            prompt=prompt.name,
            filename=pair.image.name,
            e2e_ms=elapsed_ms,
            prompt_ms=_number(timings.get("prompt_ms")),
            generation_ms=_number(timings.get("predicted_ms")),
            prompt_tokens=_integer(timings.get("prompt_n")),
            cached_tokens=_integer(timings.get("cache_n")),
            generated_tokens=_integer(timings.get("predicted_n")),
            prompt_tokens_per_second=_number(timings.get("prompt_per_second")),
            generation_tokens_per_second=_number(timings.get("predicted_per_second")),
            response=_response_text(data),
        )
    except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = str(exc)
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text.strip().replace("\n", " ")[:300]
            if body:
                detail = f"{detail}: {body}"
        return Sample(
            host=server.name,
            prompt=prompt.name,
            filename=pair.image.name,
            e2e_ms=elapsed_ms,
            error=detail,
        )


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile, including for one value."""
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "stddev": statistics.pstdev(values),
    }


def _format_seconds(value_ms: float) -> str:
    return f"{value_ms / 1000:.3f}"


def print_summary(samples: list[Sample]) -> None:
    print("\nLatency summary (seconds)")
    headings = (
        "host",
        "prompt",
        "metric",
        "n",
        "min",
        "mean",
        "p50",
        "p95",
        "max",
        "sd",
    )
    print(
        f"{headings[0]:<11} {headings[1]:<13} {headings[2]:<12} {headings[3]:>5} "
        f"{headings[4]:>9} {headings[5]:>9} {headings[6]:>9} {headings[7]:>9} "
        f"{headings[8]:>9} {headings[9]:>9}"
    )
    metrics = (
        ("prompt", lambda sample: sample.prompt_ms),
        ("generation", lambda sample: sample.generation_ms),
        ("end-to-end", lambda sample: sample.e2e_ms),
        ("unattributed", lambda sample: sample.overhead_ms),
    )
    for server in SERVERS:
        for prompt_name in ("quick_reject", "prompt_nbc"):
            group = [
                sample
                for sample in samples
                if sample.host == server.name
                and sample.prompt == prompt_name
                and sample.error is None
            ]
            for metric_name, getter in metrics:
                values = [
                    value for sample in group if (value := getter(sample)) is not None
                ]
                if not values:
                    print(
                        f"{server.name:<11} {prompt_name:<13} {metric_name:<12} {0:>5} "
                        f"{'-':>9} {'-':>9} {'-':>9} {'-':>9} {'-':>9} {'-':>9}"
                    )
                    continue
                stats = describe(values)
                print(
                    f"{server.name:<11} {prompt_name:<13} {metric_name:<12} {len(values):>5} "
                    f"{_format_seconds(stats['min']):>9} "
                    f"{_format_seconds(stats['mean']):>9} "
                    f"{_format_seconds(stats['p50']):>9} "
                    f"{_format_seconds(stats['p95']):>9} "
                    f"{_format_seconds(stats['max']):>9} "
                    f"{_format_seconds(stats['stddev']):>9}"
                )

    print("\nRequest and token summary")
    print(
        f"{'host':<11} {'prompt':<13} {'ok':>5} {'fail':>5} {'timed':>6} "
        f"{'cache%':>8} {'prompt tok/s':>14} {'gen tok/s':>11} {'gen tok':>9}"
    )
    for server in SERVERS:
        for prompt_name in ("quick_reject", "prompt_nbc"):
            group = [
                sample
                for sample in samples
                if sample.host == server.name and sample.prompt == prompt_name
            ]
            successful = [sample for sample in group if sample.error is None]
            failed = len(group) - len(successful)
            timed = sum(
                sample.prompt_ms is not None and sample.generation_ms is not None
                for sample in successful
            )
            prompt_tps = [
                value
                for sample in successful
                if (value := sample.prompt_tokens_per_second) is not None
            ]
            generation_tps = [
                value
                for sample in successful
                if (value := sample.generation_tokens_per_second) is not None
            ]
            generated = [
                value
                for sample in successful
                if (value := sample.generated_tokens) is not None
            ]
            processed = sum(sample.prompt_tokens or 0 for sample in successful)
            cached = sum(sample.cached_tokens or 0 for sample in successful)
            total_prompt = processed + cached
            cache_percent = 100 * cached / total_prompt if total_prompt else None

            def mean_or_dash(values: list[float] | list[int]) -> str:
                return f"{statistics.fmean(values):.1f}" if values else "-"

            print(
                f"{server.name:<11} {prompt_name:<13} {len(successful):>5} {failed:>5} "
                f"{timed:>6} "
                f"{(f'{cache_percent:.1f}' if cache_percent is not None else '-'):>8} "
                f"{mean_or_dash(prompt_tps):>14} {mean_or_dash(generation_tps):>11} "
                f"{mean_or_dash(generated):>9}"
            )

    errors = [sample for sample in samples if sample.error is not None]
    if errors:
        print(f"\nFailures ({len(errors)})")
        for sample in errors:
            print(f"  {sample.host}/{sample.prompt}/{sample.filename}: {sample.error}")


def _print_sample(sample: Sample, index: int, total: int) -> None:
    if sample.error is not None:
        print(
            f"[{index:>{len(str(total))}}/{total}] {sample.host:<9} "
            f"{sample.prompt:<12} ERROR {sample.error}",
            flush=True,
        )
        return
    prompt = (
        f"{sample.prompt_ms / 1000:.3f}s" if sample.prompt_ms is not None else "n/a"
    )
    generation = (
        f"{sample.generation_ms / 1000:.3f}s"
        if sample.generation_ms is not None
        else "n/a"
    )
    reply = " ".join((sample.response or "").split())[:60]
    print(
        f"[{index:>{len(str(total))}}/{total}] {sample.host:<9} "
        f"{sample.prompt:<12} e2e={sample.e2e_ms / 1000:.3f}s "
        f"prompt={prompt} gen={generation} reply={reply!r}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        pairs, missing_audio = discover_pairs(args.broadcast_dir, args.limit)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if not pairs:
        print("Error: no image+audio pairs found", file=sys.stderr)
        return 2

    prompts = (
        PromptCase("quick_reject", AUDIO_QUICK_REJECT_PROMPT, 10),
        PromptCase("prompt_nbc", load_prompt("prompt_nbc.txt"), 500),
    )
    total_requests = len(pairs) * len(SERVERS) * len(prompts)
    print(
        f"Found {len(pairs):,} image+audio pairs; running {total_requests:,} requests."
    )
    if missing_audio:
        print(f"Skipped {missing_audio:,} images without matching WAV audio.")
    for server in SERVERS:
        print(f"  {server.name:<9} {server.url}  model={server.model}")

    output_file = None
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output_file = args.output.open("x", encoding="utf-8")
        except FileExistsError:
            print(f"Error: output already exists: {args.output}", file=sys.stderr)
            return 2

    samples: list[Sample] = []
    request_index = 0
    interrupted = False
    try:
        with httpx.Client(
            timeout=httpx.Timeout(args.timeout),
            headers={"Authorization": "Bearer none"},
        ) as client:
            for pair in pairs:
                image_b64 = load_image_b64(str(pair.image))
                audio_b64 = base64.b64encode(pair.audio.read_bytes()).decode("ascii")
                for server in SERVERS:
                    for prompt in prompts:
                        request_index += 1
                        sample = run_request(
                            client,
                            server,
                            prompt,
                            pair,
                            image_b64,
                            audio_b64,
                        )
                        samples.append(sample)
                        _print_sample(sample, request_index, total_requests)
                        if output_file is not None:
                            record = asdict(sample)
                            record["overhead_ms"] = sample.overhead_ms
                            output_file.write(json.dumps(record) + "\n")
                            output_file.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted; reporting completed requests.", file=sys.stderr)
    finally:
        if output_file is not None:
            output_file.close()

    print_summary(samples)
    if interrupted:
        return 130
    return 1 if any(sample.error is not None for sample in samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
