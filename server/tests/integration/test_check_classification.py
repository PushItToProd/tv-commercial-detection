"""Integration test for check_classification.py.

Marked with @pytest.mark.integration — skipped by default.
Run with:  uv run pytest -m integration

Requirements:
- A live llama.cpp server reachable at LLAMA_SERVER_URL
- A directory of labeled frames at TEST_FRAMES_DIR (set via env var)
  - Must contain a labels.json in the same format as server/frames/labels.json
- Set ACCURACY_THRESHOLD (float, default 0.80) to fail if accuracy drops below it
- Optionally set ACCURACY_THRESHOLD_AD / ACCURACY_THRESHOLD_CONTENT for per-label thresholds

Example:
    TEST_FRAMES_DIR=/path/to/frames uv run pytest -m integration -v
"""

import json
import os
import urllib.request
from pathlib import Path

import pytest

from tv_commercial_detector.classify import classify_image

# ---------------------------------------------------------------------------
# Skip early if the environment is not set up for integration tests
# ---------------------------------------------------------------------------


def _llama_reachable() -> bool:
    """Return True if the llama.cpp server is reachable."""
    url = os.environ.get("LLAMA_SERVER_URL", "http://192.168.1.27:3002")
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _frames_dir() -> Path | None:
    raw = os.environ.get("TEST_FRAMES_DIR")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_dir() else None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frames_dir():
    d = _frames_dir()
    if d is None:
        pytest.skip(
            "TEST_FRAMES_DIR not set or not a directory — skipping integration test"
        )
    return d


@pytest.fixture(scope="module")
def labels(frames_dir: Path) -> dict:
    labels_file = frames_dir / "labels.json"
    if not labels_file.exists():
        pytest.skip(f"labels.json not found in {frames_dir}")
    with labels_file.open() as f:
        return json.load(f)


@pytest.fixture(scope="module", autouse=False)
def llama_server():
    if not _llama_reachable():
        pytest.skip("llama.cpp server not reachable — skipping integration test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_image_files(base_dir: Path) -> list[Path]:
    files = list(base_dir.glob("*.png")) + list(base_dir.glob("*.jpg"))
    return sorted(f for f in files if not f.name.startswith("compressed_"))


def _classify_all(
    image_files: list[Path], labels: dict
) -> dict[str, list[str]]:
    """Run classify_image on all labeled files.  Returns per-label result buckets."""
    buckets: dict[str, list[str]] = {
        "correct": [],
        "incorrect_ad": [],
        "incorrect_content": [],
        "unknown": [],
        "unlabeled": [],
        "ignored": [],
    }

    for f in image_files:
        actual = labels.get(f.name)
        if actual is None:
            buckets["unlabeled"].append(f.name)
            continue
        if actual == "ignore":
            buckets["ignored"].append(f.name)
            continue

        result = classify_image(f.as_posix())
        predicted = result.type

        if predicted == actual:
            buckets["correct"].append(f.name)
        elif predicted == "ad":
            buckets["incorrect_ad"].append(f.name)
        elif predicted == "content":
            buckets["incorrect_content"].append(f.name)
        else:
            buckets["unknown"].append(f.name)

    return buckets


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_overall_accuracy(frames_dir, labels, llama_server):
    """Overall accuracy across all labeled frames must exceed ACCURACY_THRESHOLD."""
    threshold = float(os.environ.get("ACCURACY_THRESHOLD", "0.80"))

    image_files = _get_image_files(frames_dir)
    buckets = _classify_all(image_files, labels)

    classified = [
        f
        for f in image_files
        if labels.get(f.name) not in (None, "ignore")
    ]
    total = len(classified)
    if total == 0:
        pytest.skip("No labeled images found — skipping accuracy assertion")

    correct = len(buckets["correct"])
    accuracy = correct / total

    print(
        f"\nAccuracy: {correct}/{total} = {accuracy:.1%}"
        f"  (incorrect_ad={len(buckets['incorrect_ad'])},"
        f" incorrect_content={len(buckets['incorrect_content'])},"
        f" unknown={len(buckets['unknown'])})"
    )
    assert accuracy >= threshold, (
        f"Accuracy {accuracy:.1%} is below threshold {threshold:.1%}.\n"
        f"Incorrectly marked as ad: {buckets['incorrect_ad']}\n"
        f"Incorrectly marked as content: {buckets['incorrect_content']}\n"
        f"Unknown: {buckets['unknown']}"
    )


@pytest.mark.integration
def test_per_label_accuracy(frames_dir, labels, llama_server):
    """Per-label accuracy must exceed configurable thresholds."""
    ad_threshold = float(os.environ.get("ACCURACY_THRESHOLD_AD", "0.75"))
    content_threshold = float(os.environ.get("ACCURACY_THRESHOLD_CONTENT", "0.75"))

    image_files = _get_image_files(frames_dir)
    buckets = _classify_all(image_files, labels)

    # Build per-label ground truth
    ad_files = [f for f in image_files if labels.get(f.name) == "ad"]
    content_files = [f for f in image_files if labels.get(f.name) == "content"]

    if ad_files:
        ad_correct = sum(
            1
            for f in ad_files
            if classify_image(f.as_posix()).type == "ad"
        )
        ad_acc = ad_correct / len(ad_files)
        print(f"\nAd accuracy: {ad_correct}/{len(ad_files)} = {ad_acc:.1%}")
        assert ad_acc >= ad_threshold, (
            f"Ad accuracy {ad_acc:.1%} below threshold {ad_threshold:.1%}"
        )

    if content_files:
        content_correct = sum(
            1
            for f in content_files
            if classify_image(f.as_posix()).type == "content"
        )
        content_acc = content_correct / len(content_files)
        print(
            f"Content accuracy: {content_correct}/{len(content_files)} = {content_acc:.1%}"
        )
        assert content_acc >= content_threshold, (
            f"Content accuracy {content_acc:.1%} below threshold {content_threshold:.1%}"
        )
