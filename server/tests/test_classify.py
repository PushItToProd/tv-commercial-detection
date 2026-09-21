"""Tests for classify_image with the OpenAI client patched.

All tests use synthetic JPEG fixtures — no real broadcast images needed.
"""

import importlib
import inspect
import io
import json
import math
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from tv_commercial_detector.classification.llm_match import (
    _extract_json,
    _get_classification_from_response,
)
from tv_commercial_detector.classification.result import ClassificationResult
from tv_commercial_detector.classify import classify_image, list_profiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(tmp_path: Path, color=(50, 50, 50), size=(64, 64)) -> str:
    """Write a small solid-colour JPEG and return its path as a string."""
    img = Image.new("RGB", size, color)
    path = tmp_path / "frame.jpg"
    img.save(str(path), format="JPEG")
    return str(path)


def _mock_openai_response(text: str):
    """Return a minimal mock that looks like an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# Every profile accepts the arguments classify.py dispatches with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", list_profiles())
def test_profile_accepts_dispatch_arguments(profile):
    """classify.py calls every profile as classify_image(image_path, audio_bytes).

    A mismatch raises TypeError inside /receive, which catches it and records
    the frame as unknown, so a broken profile never switches the matrix and
    only shows up as a log line.
    """
    module = importlib.import_module(f"tv_commercial_detector.classifiers.{profile}")
    inspect.signature(module.classify_image).bind("frame.jpg", b"")


# ---------------------------------------------------------------------------
# _get_classification_from_response — pure parsing, no I/O
# ---------------------------------------------------------------------------


def test_parse_type_equals_ad():
    result = _get_classification_from_response("type=ad some explanation")
    assert result.type == "ad"
    assert result.source == "llm"


def test_parse_type_equals_racing():
    result = _get_classification_from_response("type=racing broadcast")
    assert result.type == "content"


def test_parse_json_classification_ad():
    reply = json.dumps({"classification": "ad", "reason": "no-racing"})
    result = _get_classification_from_response(reply)
    assert result.type == "ad"


def test_parse_json_classification_racing():
    reply = json.dumps({"classification": "racing", "reason": "scoreboard visible"})
    result = _get_classification_from_response(reply)
    assert result.type == "content"


def test_parse_unknown_for_garbage():
    result = _get_classification_from_response("I have no idea what this is")
    assert result.type == "unknown"


def test_parse_empty_reply_returns_unknown():
    result = _get_classification_from_response("")
    assert result.type == "unknown"


def test_parse_json_with_unknown_classification():
    reply = json.dumps({"classification": "unsure"})
    result = _get_classification_from_response(reply)
    assert result.type == "unknown"


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_valid():
    text = 'Some preamble {"classification": "ad"} trailing'
    obj = _extract_json(text)
    assert obj == {"classification": "ad"}


def test_extract_json_no_json():
    assert _extract_json("no braces here") is None


def test_extract_json_malformed():
    assert _extract_json("{not valid json}") is None


# ---------------------------------------------------------------------------
# classify_image — LLM path mocked, OpenCV skipped via blank image
# ---------------------------------------------------------------------------


def test_classify_image_llm_returns_ad(tmp_path, mocker):
    """
    Blank JPEG → OpenCV finds no logo/rectangle → falls through to LLM.
    Mock the quick-reject to say 'yes racing-related', then mock full prompt to
    return ad.
    """
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classification.rectangle_match.image_has_known_ad_rectangle",
        return_value=None,
    )

    # Quick-reject says racing-related (so we proceed to full prompt)
    mocker.patch(
        "tv_commercial_detector.classification.llm_match._report_racing_related",
        return_value=True,
    )
    # Full prompt returns "ad"
    mocker.patch(
        "tv_commercial_detector.classification.llm_match.classify_by_prompt",
        return_value=ClassificationResult(
            source="llm", type="ad", reason="model-match", reply='{"classification":"ad"}'
        ),
    )

    result = classify_image(image_path)
    assert result.type == "ad"
    assert result.source == "llm"


def test_classify_image_llm_returns_content(tmp_path, mocker):
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classification.rectangle_match.image_has_known_ad_rectangle",
        return_value=None,
    )
    mocker.patch(
        "tv_commercial_detector.classification.llm_match._report_racing_related",
        return_value=True,
    )
    mocker.patch(
        "tv_commercial_detector.classification.llm_match.classify_by_prompt",
        return_value=ClassificationResult(
            source="llm", type="content", reason="model-match", reply='{"classification":"racing"}'
        ),
    )

    result = classify_image(image_path)
    assert result.type == "content"


def test_classify_image_quick_reject_returns_ad(tmp_path, mocker):
    """If racing-related quick check returns False, classify as ad immediately."""
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classification.rectangle_match.image_has_known_ad_rectangle",
        return_value=None,
    )
    mocker.patch(
        "tv_commercial_detector.classification.llm_match._report_racing_related",
        return_value=False,
    )
    # classify_by_prompt should NOT be called
    mock_prompt = mocker.patch("tv_commercial_detector.classification.llm_match.classify_by_prompt")

    result = classify_image(image_path)
    assert result.type == "ad"
    assert result.reason == "model_quick_reject"
    mock_prompt.assert_not_called()


def test_classify_image_opencv_network_logo_wins(tmp_path, mocker):
    """If OpenCV finds a network logo, classify as content without calling LLM."""
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=True,
    )
    mock_llm = mocker.patch("tv_commercial_detector.classification.llm_match._report_racing_related")

    result = classify_image(image_path)
    assert result.type == "content"
    assert result.source == "opencv"
    assert result.reason == "network_logo"
    mock_llm.assert_not_called()


def test_classify_image_opencv_side_by_side_wins(tmp_path, mocker):
    """Side-by-side logo detection returns ad without calling LLM."""
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=True,
    )
    mock_llm = mocker.patch("tv_commercial_detector.classification.llm_match._report_racing_related")

    result = classify_image(image_path)
    assert result.type == "ad"
    assert result.source == "opencv"
    assert result.reason == "side_by_side"
    mock_llm.assert_not_called()


def test_classify_image_opencv_rectangle_wins(tmp_path, mocker):
    """Rectangle detection returns ad without calling LLM."""
    image_path = _make_jpeg(tmp_path)

    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classification.rectangle_match.image_has_known_ad_rectangle",
        return_value="fox-side-by-side-left",
    )
    mock_llm = mocker.patch("tv_commercial_detector.classification.llm_match._report_racing_related")

    result = classify_image(image_path)
    assert result.type == "ad"
    assert result.source == "opencv"
    assert result.reason == "matched_rectangle"
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# _classify_by_prompt — OpenAI client mock (lower level)
# ---------------------------------------------------------------------------


def test_classify_by_prompt_parses_ad_json(mocker):
    """Patch OpenAI() so the network never touches the actual LLM."""
    from tv_commercial_detector.classification.llm_match import classify_by_prompt

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        '{"classification": "ad", "reason": "no scoreboard"}'
    )
    mocker.patch("tv_commercial_detector.classification.llm_match.OpenAI", return_value=mock_client)
    mocker.patch("tv_commercial_detector.classification.llm_match.CLASSIFICATION_TIME.time")

    result = classify_by_prompt("fakebase64data==")
    assert result.type == "ad"


def test_classify_by_prompt_parses_racing_json(mocker):
    from tv_commercial_detector.classification.llm_match import classify_by_prompt

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        '{"classification": "racing", "reason": "scoreboard visible"}'
    )
    mocker.patch("tv_commercial_detector.classification.llm_match.OpenAI", return_value=mock_client)
    mocker.patch("tv_commercial_detector.classification.llm_match.CLASSIFICATION_TIME.time")

    result = classify_by_prompt("fakebase64data==")
    assert result.type == "content"


def test_classify_by_prompt_handles_none_content(mocker):
    """If the LLM returns None content, return unknown."""
    from tv_commercial_detector.classification.llm_match import classify_by_prompt

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(None)
    mocker.patch("tv_commercial_detector.classification.llm_match.OpenAI", return_value=mock_client)
    mocker.patch("tv_commercial_detector.classification.llm_match.CLASSIFICATION_TIME.time")

    result = classify_by_prompt("fakebase64data==")
    assert result.type == "unknown"
    assert result.reason == "empty_response"


def test_classify_by_prompt_handles_malformed_json(mocker):
    """Malformed JSON but with a regex match should still parse correctly."""
    from tv_commercial_detector.classification.llm_match import classify_by_prompt

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(
        "type=ad (could not parse as json)"
    )
    mocker.patch("tv_commercial_detector.classification.llm_match.OpenAI", return_value=mock_client)
    mocker.patch("tv_commercial_detector.classification.llm_match.CLASSIFICATION_TIME.time")

    result = classify_by_prompt("fakebase64data==")
    assert result.type == "ad"


# ---------------------------------------------------------------------------
# Silent audio is withheld from both LLM passes
# ---------------------------------------------------------------------------


def _wav(amplitude: int, seconds: float = 0.1) -> bytes:
    t = np.arange(int(44100 * seconds)) / 44100
    samples = (amplitude * np.sin(2 * math.pi * 440 * t)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _audio_passed_to_llm(tmp_path, mocker, audio_bytes):
    """Run the LLM path of classify_image and report what audio each pass saw."""
    image_path = _make_jpeg(tmp_path)
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_network_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classifiers.nascar_on_fox.has_side_by_side_logo",
        return_value=False,
    )
    mocker.patch(
        "tv_commercial_detector.classification.rectangle_match.image_has_known_ad_rectangle",
        return_value=None,
    )
    quick = mocker.patch(
        "tv_commercial_detector.classification.llm_match._report_racing_related",
        return_value=True,
    )
    full = mocker.patch(
        "tv_commercial_detector.classification.llm_match.classify_by_prompt",
        return_value=ClassificationResult(
            source="llm", type="content", reason="model-match", reply=""
        ),
    )
    classify_image(image_path, audio_bytes)
    return quick.call_args.args[1], full.call_args.args[1]


def test_silent_audio_is_not_sent_to_either_llm_pass(tmp_path, mocker):
    """A silent clip is worse than none: the model invents engine noise from it."""
    assert _audio_passed_to_llm(tmp_path, mocker, _wav(amplitude=0)) == (None, None)


def test_audio_with_signal_is_sent_to_both_llm_passes(tmp_path, mocker):
    quick, full = _audio_passed_to_llm(tmp_path, mocker, _wav(amplitude=8000))
    assert quick is not None
    assert quick == full
