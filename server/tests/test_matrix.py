"""Matrix client contract: one fully validated batch, no fire-and-forget flag."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from tv_commercial_detector.config import app_config
from tv_commercial_detector.matrix import apply_matrix_settings


def send(settings, *, error=None):
    app_config.matrix_url = "http://matrix:11344/"
    app_config.output_settings = {"ad": settings}
    response = MagicMock()
    response.__enter__.return_value.status = 200
    with patch(
        "tv_commercial_detector.matrix.urllib.request.urlopen",
        return_value=response,
        side_effect=error,
    ) as urlopen:
        asyncio.run(apply_matrix_settings("ad"))
    return urlopen


def test_both_outputs_are_one_apply_request():
    urlopen = send({"A": "1", "B": 4, "label": "Commercials"})
    urlopen.assert_called_once()
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://matrix:11344/apply"
    assert request.method == "POST"
    assert json.loads(request.data) == {"A": 1, "B": 4}
    assert urlopen.call_args.kwargs["timeout"] == 5


def test_single_output_and_invalid_output_key(caplog):
    urlopen = send({"B": 1, "C": 2})
    assert json.loads(urlopen.call_args.args[0].data) == {"B": 1}
    assert "invalid output 'C'" in caplog.text


@pytest.mark.parametrize("settings", [{}, {"label": "Empty"}, {"C": 1}])
def test_no_routes_means_no_request(settings):
    send(settings).assert_not_called()


@pytest.mark.parametrize("bad", [0, 5, True, 1.5, "bad", None])
def test_bad_input_cannot_cause_partial_application(bad):
    send({"A": 1, "B": bad}).assert_not_called()


def test_timeout_is_logged_without_replaying(caplog):
    urlopen = send({"A": 1, "B": 4}, error=TimeoutError("reply lost"))
    urlopen.assert_called_once()
    assert "Matrix error" in caplog.text
