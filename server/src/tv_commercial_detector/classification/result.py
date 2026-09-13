from dataclasses import dataclass


@dataclass
class ClassificationResult:
    source: str
    type: str  # "ad", "content", or "unknown"
    reason: str
    reply: str | None
    # Raw scores behind the verdict, recorded whatever decided it — e.g.
    # `p_audio` — so a later policy can weigh evidence a verdict throws away.
    signals: dict | None = None
