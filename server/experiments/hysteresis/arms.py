"""Temporal decision policies, and the harness that scores them.

Every policy consumes the same stream of per-frame classifier outputs and emits
the state the HDMI matrix would be switched to. They differ only in how much
history they keep and what they do with it, so a difference in score is a
difference in the policy rather than in the underlying classifier.

Episodes are the unit of simulation: a run of consecutive archived frames at the
live ~2 s cadence. The archive keeps only such runs (`recent_frames` is a
5-deep deque flushed once a minute), so nothing longer can be replayed
faithfully. Each episode starts from the ground-truth state of the last labelled
frame before it, which is neutral across policies - no policy gets to carry in
its own earlier mistakes, and none is punished for another's.

Cost model: OpenCV runs on every frame regardless. Anything a policy skips is
the LLM pass, which is the only part that costs more than single-digit
milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Measured on this machine over the race window (see extract_features.py):
# the full OpenCV pass - resize to 1080p plus the peacock, USA and side-by-side
# matches - has a median of 2.4 ms and a p95 of 3.8 ms.
OPENCV_SECS = 0.0024
PHASH_SECS = 0.0047


@dataclass
class Frame:
    """One archived frame and everything a policy may look at."""
    index: int
    episode: int
    seconds: float          # wall clock, seconds since window start
    gt: str                 # ad | content | uncertain
    verdict: str            # per-frame classifier output for this rep
    reason: str             # side_by_side | network_logo | model_quick_reject | model-match
    source: str             # opencv | llm
    llm_secs: float         # measured cost of the LLM passes, 0 when short-circuited
    peacock: float
    usa: float
    sbs: float
    phash_dist_prev: int | None


@dataclass
class Decision:
    state: str
    secs: float             # simulated wall time for this frame
    called_llm: bool


class Policy:
    """Base class. `reset` starts an episode, `step` handles one frame."""
    name = "base"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state

    def step(self, f: Frame) -> Decision:
        raise NotImplementedError

    # Cost of running the classifier on this frame, as the pipeline would.
    @staticmethod
    def _full_cost(f: Frame) -> float:
        return OPENCV_SECS + f.llm_secs


class Stateless(Policy):
    """Current behaviour with debounce off: emit whatever the frame says."""
    name = "stateless"

    def step(self, f: Frame) -> Decision:
        if f.verdict in ("ad", "content"):
            self.state = f.verdict
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


class ProductionDebounce(Policy):
    """What `receive.py` does today.

    An OpenCV verdict commits immediately; an LLM verdict has to appear twice in
    a row before it moves the state. `unknown` is ignored rather than breaking
    the run, matching the comment in receive.py.
    """
    name = "debounce2"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state
        self.prev: str | None = None

    def step(self, f: Frame) -> Decision:
        result = f.verdict
        prev = self.prev
        if result != "unknown":
            self.prev = result
        if result in ("ad", "content") and result != self.state:
            if f.source == "opencv" or result == prev:
                self.state = result
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


class MajorityK(Policy):
    """Majority vote over the last k per-frame verdicts."""

    def __init__(self, k: int):
        self.k = k
        self.name = f"majority{k}"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state
        self.window: list[str] = []

    def step(self, f: Frame) -> Decision:
        if f.verdict in ("ad", "content"):
            self.window.append(f.verdict)
            self.window = self.window[-self.k :]
        if self.window:
            ads = self.window.count("ad")
            if ads * 2 > len(self.window):
                self.state = "ad"
            elif ads * 2 < len(self.window):
                self.state = "content"
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


class Hysteresis(Policy):
    """Asymmetric consecutive-run counters.

    `n_enter` consecutive `ad` verdicts to switch into ad, `n_exit` consecutive
    `content` verdicts to come back. Asymmetry is the point: entering an ad
    break late costs a few seconds of commercial, but leaving late costs live
    racing, so the two directions should not need the same evidence.
    """

    def __init__(self, n_enter: int, n_exit: int, opencv_immediate: bool = False):
        self.n_enter, self.n_exit = n_enter, n_exit
        self.opencv_immediate = opencv_immediate
        suffix = "+cv" if opencv_immediate else ""
        self.name = f"hyst{n_enter}/{n_exit}{suffix}"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state
        self.run_label: str | None = None
        self.run_len = 0

    def step(self, f: Frame) -> Decision:
        v = f.verdict
        if v in ("ad", "content"):
            if v == self.run_label:
                self.run_len += 1
            else:
                self.run_label, self.run_len = v, 1

            # A side-by-side or network-bug match is a direct observation of
            # broadcast furniture, not an opinion, so it may commit at once.
            if self.opencv_immediate and f.source == "opencv":
                self.state = v
            else:
                need = self.n_enter if v == "ad" else self.n_exit
                if self.run_len >= need:
                    self.state = v
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


class LogOdds(Policy):
    """Evidence accumulator with per-source weights and decay.

    Counters treat every verdict as equally good. They are not: a side-by-side
    banner match was measured at ~0.18 of margin, while an LLM verdict on a
    cinematic car commercial is close to a coin flip. This keeps a running score
    - positive means ad - adding a weight per frame, decaying toward zero so
    stale evidence fades, and switching on asymmetric thresholds.
    """

    WEIGHTS = {
        "side_by_side": 3.0,
        "network_logo": -3.0,
        "model_quick_reject": 0.8,
        "model-match": 1.0,
    }

    def __init__(self, enter: float = 2.0, exit_: float = 1.5, decay: float = 0.7):
        self.enter, self.exit_, self.decay = enter, exit_, decay
        self.name = f"logodds{enter:g}/{exit_:g}/d{decay:g}"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state
        self.score = self.enter if initial_state == "ad" else -self.exit_

    def step(self, f: Frame) -> Decision:
        w = self.WEIGHTS.get(f.reason, 1.0)
        if f.reason in ("side_by_side", "network_logo"):
            delta = w                      # sign is carried by the weight
        elif f.verdict == "ad":
            delta = w
        elif f.verdict == "content":
            delta = -w
        else:
            delta = 0.0
        self.score = self.score * self.decay + delta

        if self.state != "ad" and self.score >= self.enter:
            self.state = "ad"
        elif self.state == "ad" and self.score <= -self.exit_:
            self.state = "content"
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


class DuplicateCache(Policy):
    """Skip the LLM when the frame is a near-duplicate of the previous one.

    At a 2 s cadence a great many consecutive frames are the same shot, and the
    classifier is being asked the same question again at full price. A
    perceptual-hash distance under `max_dist` reuses the previous verdict for
    the cost of a hash. The decision that follows is delegated to `inner`, so
    this composes with any policy above.
    """

    def __init__(self, inner: Policy, max_dist: int = 4):
        self.inner = inner
        self.max_dist = max_dist
        self.name = f"cache{max_dist}+{inner.name}"

    @property
    def state(self) -> str:
        return self.inner.state

    def reset(self, initial_state: str) -> None:
        self.inner.reset(initial_state)
        self.last_verdict: str | None = None
        self.last_reason: str | None = None
        self.last_source: str | None = None

    def step(self, f: Frame) -> Decision:
        d = f.phash_dist_prev
        reuse = (
            self.last_verdict is not None
            and d is not None
            and d <= self.max_dist
        )
        if reuse:
            g = Frame(**{**f.__dict__,
                         "verdict": self.last_verdict,
                         "reason": self.last_reason,
                         "source": self.last_source,
                         "llm_secs": 0.0})
            dec = self.inner.step(g)
            # OpenCV still runs; the hash replaces only the LLM passes.
            return Decision(dec.state, OPENCV_SECS + PHASH_SECS, False)

        dec = self.inner.step(f)
        self.last_verdict, self.last_reason = f.verdict, f.reason
        self.last_source = f.source
        return Decision(dec.state, dec.secs + PHASH_SECS, f.llm_secs > 0)


class BugMemory(Policy):
    """Hysteresis that also remembers when broadcast furniture was last seen.

    The strongest cue in the hard cases is a negative one: live coverage almost
    never runs many seconds with no network bug at all, and commercials always
    do. This tracks how long since the upper-right bug last matched and uses it
    to bias the switch into ad - fast when the bug has been gone a while, slow
    while it was on screen moments ago. That is the signal the per-frame
    classifier cannot see, and it is free, because the bug scores are already
    computed on every frame.
    """

    def __init__(self, n_enter: int = 3, n_exit: int = 2, recent_secs: float = 6.0):
        self.n_enter, self.n_exit, self.recent_secs = n_enter, n_exit, recent_secs
        self.name = f"bugmem{n_enter}/{n_exit}/{recent_secs:g}s"

    def reset(self, initial_state: str) -> None:
        self.state = initial_state
        self.run_label: str | None = None
        self.run_len = 0
        self.last_bug_secs: float | None = None

    def step(self, f: Frame) -> Decision:
        bug = f.peacock >= 0.55 or f.usa >= 0.65
        if bug:
            self.last_bug_secs = f.seconds

        v = f.verdict
        if v in ("ad", "content"):
            if v == self.run_label:
                self.run_len += 1
            else:
                self.run_label, self.run_len = v, 1

            if f.source == "opencv":
                self.state = v
            elif v == "ad":
                # Demand more evidence while the bug was on screen recently.
                seen_recently = (
                    self.last_bug_secs is not None
                    and f.seconds - self.last_bug_secs <= self.recent_secs
                )
                need = self.n_enter + 1 if seen_recently else self.n_enter
                if self.run_len >= need:
                    self.state = "ad"
            else:
                if self.run_len >= self.n_exit:
                    self.state = "content"
        return Decision(self.state, self._full_cost(f), f.llm_secs > 0)


@dataclass
class Score:
    frames: int = 0
    correct: int = 0
    ad_total: int = 0
    ad_correct: int = 0
    content_total: int = 0
    content_correct: int = 0
    switches: int = 0
    gt_changes: int = 0
    llm_calls: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.frames if self.frames else 0.0

    @property
    def balanced(self) -> float:
        a = self.ad_correct / self.ad_total if self.ad_total else 0.0
        c = self.content_correct / self.content_total if self.content_total else 0.0
        return (a + c) / 2
