from dataclasses import dataclass


@dataclass
class ClassificationResult:
    source: str
    type: str  # "ad", "content", or "unknown"
    reason: str
    reply: str | None
