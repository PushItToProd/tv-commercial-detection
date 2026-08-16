import json
from pathlib import Path

import httpx
import pytest
from scripts import benchmark_qwen3_omni as benchmark


def test_discover_pairs_matches_stems_and_applies_limit(tmp_path: Path):
    images = tmp_path / "images"
    audio = tmp_path / "audio"
    images.mkdir()
    audio.mkdir()
    for name in ("b.jpg", "a.jpg", "missing.jpg"):
        (images / name).write_bytes(b"image")
    for name in ("a.wav", "b.wav"):
        (audio / name).write_bytes(b"audio")

    pairs, missing = benchmark.discover_pairs(tmp_path, limit=1)

    assert [pair.image.name for pair in pairs] == ["a.jpg"]
    # Discovery stops once the requested number of valid pairs is found.
    assert missing == 0


def test_discover_pairs_reports_images_without_audio(tmp_path: Path):
    images = tmp_path / "images"
    audio = tmp_path / "audio"
    images.mkdir()
    audio.mkdir()
    (images / "a.jpg").write_bytes(b"image")
    (images / "b.jpg").write_bytes(b"image")
    (audio / "b.wav").write_bytes(b"audio")

    pairs, missing = benchmark.discover_pairs(tmp_path)

    assert [pair.image.name for pair in pairs] == ["b.jpg"]
    assert missing == 1


def test_build_payload_uses_router_model_and_both_media_inputs():
    server = benchmark.Server("router", "http://router", benchmark.ROUTER_MODEL)
    prompt = benchmark.PromptCase("quick", "question", 10)

    payload = benchmark.build_payload(server, prompt, "image-data", "audio-data")

    assert payload["model"] == benchmark.ROUTER_MODEL
    assert payload["max_tokens"] == 10
    content = payload["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "question"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,image-data"},
        },
        {
            "type": "input_audio",
            "input_audio": {"data": "audio-data", "format": "wav"},
        },
    ]


def test_run_request_records_llama_timings_and_wall_time(mocker, tmp_path: Path):
    response_data = {
        "choices": [{"message": {"content": "Yes"}}],
        "timings": {
            "cache_n": 100,
            "prompt_n": 20,
            "prompt_ms": 125.5,
            "prompt_per_second": 159.36,
            "predicted_n": 2,
            "predicted_ms": 50.25,
            "predicted_per_second": 39.8,
        },
    }
    request = httpx.Request("POST", "http://server/v1/chat/completions")
    response = httpx.Response(200, request=request, json=response_data)
    client = mocker.Mock()
    client.post.return_value = response
    mocker.patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.25])
    pair = benchmark.MediaPair(tmp_path / "frame.jpg", tmp_path / "frame.wav")

    sample = benchmark.run_request(
        client,
        benchmark.Server("server", "http://server", "local"),
        benchmark.PromptCase("quick", "question", 10),
        pair,
        "image",
        "audio",
    )

    assert sample.error is None
    assert sample.e2e_ms == 250
    assert sample.prompt_ms == 125.5
    assert sample.generation_ms == 50.25
    assert sample.prompt_tokens == 20
    assert sample.cached_tokens == 100
    assert sample.generated_tokens == 2
    assert sample.overhead_ms == 74.25
    assert sample.response == "Yes"


def test_run_request_preserves_http_error(mocker, tmp_path: Path):
    request = httpx.Request("POST", "http://server/v1/chat/completions")
    response = httpx.Response(
        500, request=request, content=json.dumps({"error": "model failed"})
    )
    client = mocker.Mock()
    client.post.return_value = response
    mocker.patch.object(benchmark.time, "perf_counter", side_effect=[10.0, 10.1, 10.1])
    pair = benchmark.MediaPair(tmp_path / "frame.jpg", tmp_path / "frame.wav")

    sample = benchmark.run_request(
        client,
        benchmark.Server("server", "http://server", "local"),
        benchmark.PromptCase("quick", "question", 10),
        pair,
        "image",
        "audio",
    )

    assert sample.error is not None
    assert "model failed" in sample.error


def test_describe_uses_interpolated_percentiles():
    stats = benchmark.describe([1.0, 2.0, 3.0, 4.0])

    assert stats["mean"] == 2.5
    assert stats["p50"] == 2.5
    assert stats["p95"] == pytest.approx(3.85)
