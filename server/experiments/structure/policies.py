"""Temporal policies, including the duration-aware state machine.

Every policy is strictly causal - it sees frames 0..i to decide frame i - because
the thing being driven is an HDMI switch, and a policy that peeks ahead cannot
be shipped.

`stateless` and `debounce2` are the two that exist today. `hsmm` is the one this
experiment is about.

The idea behind `hsmm` is that the broadcast's own structure is evidence. A
break is never short: the shortest of the seventeen here is 118 s, and the six
NASCAR NON STOP breaks are 120 s to within a frame. A content run between two
breaks is never short either - the shortest is 94 s. So a "this looks like
content" vote arriving 20 s into a commercial break is not weak evidence to be
averaged in, it is evidence about something that has never once happened, and
the right thing to do with it is discard it.

That is what a fixed debounce cannot express. Debounce spends the same two
frames of scepticism everywhere: too much at a real boundary, where every frame
of lag is another two seconds of commercial on screen, and nowhere near enough
in the middle of a break. Making the scepticism a function of how long the
current state has held gives both back at once - immovable early, immediately
responsive once a break has run its expected length.

Past the minimum dwell, evidence accumulates as a one-sided sequential test
(CUSUM) rather than a fixed count, so a decisive frame switches at once and a
marginal one has to be sustained.
"""

import math

# Frames, at a ~2 s cadence.
NONSTOP_LEN = 60  # a NASCAR NON STOP break is exactly this long
MIN_AD_DWELL = 50  # 100 s; shortest break observed is 118 s
MIN_CONTENT_DWELL = 40  # 80 s; shortest interior content run is 94 s
CUSUM_TH = 2.0  # nats of accumulated evidence needed to switch
CLAMP = 1.5  # per-frame log-odds cap, so one wild frame cannot switch


def _logodds(p, eps=1e-4):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def stateless(ev, key):
    """Emit each frame's own verdict. What runs with debounce off."""
    out = []
    for e in ev:
        if e["anchor"] is not None:
            out.append(e["anchor"])
        elif key is None:
            out.append("content")
        else:
            out.append("ad" if e[key] >= 0.5 else "content")
    return out


def debounce2(ev, key):
    """Anchors commit at once; a sensor verdict must repeat before it moves."""
    out, state, pending, n = [], "content", None, 0
    for e in ev:
        if e["anchor"] is not None:
            state, pending, n = e["anchor"], None, 0
        else:
            v = "content" if key is None else ("ad" if e[key] >= 0.5 else "content")
            if v != state:
                if v == pending:
                    n += 1
                else:
                    pending, n = v, 1
                if n >= 2:
                    state, pending, n = v, None, 0
            else:
                pending, n = None, 0
        out.append(state)
    return out


def hsmm(
    ev,
    key,
    nonstop_len=NONSTOP_LEN,
    min_ad=MIN_AD_DWELL,
    min_content=MIN_CONTENT_DWELL,
    cusum_th=CUSUM_TH,
):
    """Duration-aware state machine. See module docstring."""
    out = []
    state, kind, dwell, acc = "content", None, 10**6, 0.0

    for e in ev:
        dwell += 1
        # --- forced exit: a NON STOP break has a known, fixed length ----------
        if state == "ad" and kind == "nonstop":
            if dwell >= nonstop_len and not e["banner"]:
                state, kind, dwell, acc = "content", None, 0, 0.0
            out.append(state)
            continue

        # --- entering a NON STOP break is unambiguous -------------------------
        if e["banner"] and state != "ad":
            state, kind, dwell, acc = "ad", "nonstop", 0, 0.0
            out.append(state)
            continue

        # --- a near-black frame only ever happens inside a break --------------
        if e["black"] and state == "content" and dwell >= min_content:
            state, kind, dwell, acc = "ad", "full", 0, 0.0
            out.append(state)
            continue

        floor = min_ad if state == "ad" else min_content
        if dwell < floor:
            acc = 0.0  # inside the dwell floor, contrary evidence is noise
            out.append(state)
            continue

        # --- past the floor: accumulate evidence for leaving ------------------
        if e["anchor"] is not None:
            ll = 8.0 if e["anchor"] == "ad" else -8.0
        elif key is None:
            ll = -1.0
        else:
            ll = _logodds(e[key])
        ll = max(-CLAMP, min(CLAMP, ll))
        toward_switch = ll if state == "content" else -ll
        acc = max(0.0, acc + toward_switch)
        if acc >= cusum_th:
            state = "ad" if state == "content" else "content"
            kind = "full" if state == "ad" else None
            dwell, acc = 0, 0.0
        out.append(state)
    return out


POLICIES = {"stateless": stateless, "debounce2": debounce2, "hsmm": hsmm}


def debounceN(ev, key, n=2):
    """Debounce with an arbitrary repeat count, for the ablation.

    The question this answers: is the state machine winning because it is a
    state machine, or only because it is more sceptical than a 2-frame
    debounce? Turning the count up is the cheap version of the same idea.
    """
    out, state, pending, k = [], "content", None, 0
    for e in ev:
        if e["anchor"] is not None:
            state, pending, k = e["anchor"], None, 0
        else:
            v = "content" if key is None else ("ad" if e[key] >= 0.5 else "content")
            if v != state:
                k = k + 1 if v == pending else 1
                pending = v
                if k >= n:
                    state, pending, k = v, None, 0
            else:
                pending, k = None, 0
        out.append(state)
    return out
