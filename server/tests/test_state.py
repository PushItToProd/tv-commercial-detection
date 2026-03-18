"""Tests for AppState — transitions, debounce, and SSE queue helpers."""

import asyncio
import time

from tv_commercial_detector.state import AppState


def test_defaults():
    s = AppState()
    assert s.classification is None
    assert s.paused is True
    assert s.seeking is False
    assert s.auto_switch is True
    assert s.enable_debounce is True
    assert s.last_result is None
    assert s.matrix_switching is False
    assert s.auto_switch_paused_until is None


# ---------------------------------------------------------------------------
# is_pending_change
# ---------------------------------------------------------------------------


def test_pending_change_no_last_result():
    s = AppState()
    s.last_result = None
    s.classification = None
    assert s.is_pending_change() is False


def test_pending_change_matches_classification():
    """last_result == classification → no pending change."""
    s = AppState()
    s.last_result = "ad"
    s.classification = "ad"
    assert s.is_pending_change() is False


def test_pending_change_differs_from_classification():
    """last_result != classification → pending change."""
    s = AppState()
    s.last_result = "content"
    s.classification = "ad"
    assert s.is_pending_change() is True


def test_pending_change_none_to_ad():
    s = AppState()
    s.last_result = "ad"
    s.classification = None
    assert s.is_pending_change() is True


# ---------------------------------------------------------------------------
# is_auto_switch_paused
# ---------------------------------------------------------------------------


def test_auto_switch_not_paused_by_default():
    s = AppState()
    assert s.is_auto_switch_paused() is False


def test_auto_switch_paused_future():
    s = AppState()
    s.auto_switch_paused_until = time.time() + 60
    assert s.is_auto_switch_paused() is True


def test_auto_switch_paused_expired():
    s = AppState()
    s.auto_switch_paused_until = time.time() - 1
    assert s.is_auto_switch_paused() is False


def test_auto_switch_paused_none():
    s = AppState()
    s.auto_switch_paused_until = None
    assert s.is_auto_switch_paused() is False


# ---------------------------------------------------------------------------
# Debounce logic (simulated — mirrors routes/receive.py logic)
# ---------------------------------------------------------------------------


def _apply_debounce(state: AppState, result: str) -> bool:
    """Return True if the new classification should be committed.

    Mirrors the condition in routes/receive.py.
    """
    if result not in ("ad", "content"):
        return False
    prev = state.last_result
    if result != "unknown":
        state.last_result = result

    source = "llm"  # non-opencv path
    return (
        source == "opencv"
        or result == prev
        or not state.enable_debounce
    ) and result != state.classification


def test_debounce_disabled_commits_immediately():
    s = AppState()
    s.enable_debounce = False
    s.classification = None
    committed = _apply_debounce(s, "ad")
    assert committed is True


def test_debounce_enabled_requires_two_consecutive():
    """With debounce on, first occurrence is not committed."""
    s = AppState()
    s.enable_debounce = True
    s.classification = None

    committed_first = _apply_debounce(s, "ad")
    assert committed_first is False  # first "ad" not yet confirmed

    committed_second = _apply_debounce(s, "ad")
    assert committed_second is True  # second consecutive "ad" commits


def test_debounce_resets_on_alternating_results():
    """Alternating results should never commit with debounce on."""
    s = AppState()
    s.enable_debounce = True
    s.classification = None

    _apply_debounce(s, "ad")
    committed = _apply_debounce(s, "content")
    assert committed is False  # never two consecutive same results


def test_no_commit_when_result_matches_current_classification():
    s = AppState()
    s.enable_debounce = False
    s.classification = "ad"
    s.last_result = "ad"

    committed = _apply_debounce(s, "ad")
    assert committed is False
