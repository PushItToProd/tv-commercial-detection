import base64
import io
import json
import os
import re
from pathlib import Path

import prometheus_client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from ..config import app_config
from .result import ClassificationResult

# Model name is now read from app_config.llm_model_name at call time.

# Reuse a single client to avoid leaking sockets (httpx connection pool).
# Lazily initialize to keep tests deterministic when OpenAI is patched.
_client: OpenAI | None = None
_client_factory = OpenAI
_client_url: str | None = None


def _get_client() -> OpenAI:
    global _client, _client_factory, _client_url
    current_url = app_config.llm_url
    if _client is None or _client_factory is not OpenAI or _client_url != current_url:
        _client = OpenAI(base_url=f"{current_url}/v1", api_key="none")
        _client_factory = OpenAI
        _client_url = current_url
    return _client

PROMPT_DIR = Path(__file__).parent.parent / "prompt"
PROMPT_FILE = os.environ.get("PROMPT_FILE", PROMPT_DIR / "prompt.txt")

# TODO: Support multiple prompts. This prompt is hardcoded and assumes Fox
# specifically, so it's probably less accurate for other TV networks.
PROMPT = Path(PROMPT_FILE).read_text()

MAX_DIMENSION = 800

CLASSIFICATION_TIME = prometheus_client.Histogram(
    "classification_time_seconds",
    "Time spent classifying each image",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 5, 10],
)

AD_MATCH_REGEX = re.compile(r"\btype=ad\b|\"classification\"\s*:\s*\"ad\"")
RACING_MATCH_REGEX = re.compile(r"\btype=racing\b|\"classification\"\s*:\s*\"racing\"")


def _resize_image(image_path: str) -> bytes:
    """Resize image so its longest side is at most MAX_DIMENSION,
    then return JPEG bytes."""
    with Image.open(image_path) as img:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()


def load_image_b64(image_path: str) -> str:
    image_data = base64.b64encode(_resize_image(image_path)).decode("utf-8")
    return image_data


def _extract_json(reply: str) -> dict | None:
    """Extract and parse the first complete JSON object from the reply."""
    start = reply.find("{")
    end = reply.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(reply[start : end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _get_classification_from_response(reply: str) -> ClassificationResult:
    """Interpret a raw LLM reply.

    Parsed via regex first, then JSON fallback.
    """
    if AD_MATCH_REGEX.search(reply):
        return ClassificationResult(
            source="llm", type="ad", reason="model-match", reply=reply
        )
    if RACING_MATCH_REGEX.search(reply):
        return ClassificationResult(
            source="llm", type="content", reason="model-match", reply=reply
        )

    data = _extract_json(reply)
    if data is not None:
        classification = data.get("classification")
        if classification == "racing":
            return ClassificationResult(
                source="llm", type="content", reason="model-match", reply=reply
            )
        if classification == "ad":
            return ClassificationResult(
                source="llm", type="ad", reason="model-match", reply=reply
            )

    return ClassificationResult(
        source="llm", type="unknown", reason="model-match", reply=reply
    )


def _report_racing_related(image_data: str, audio_data: str | None = None) -> bool:
    """Ask the LLM whether the image contains NASCAR racing content."""
    msg_content: list = [
        {
            "type": "text",
            "text": (
                "Does this image contain anything related to NASCAR racing?"
                " Reply with only 'yes' or 'no'."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ]
    if audio_data is not None and app_config.enable_llm_audio:
        msg_content[0]["text"] = (
            "This image and audio clip are from the same segment of a video. "
            "Based on both the audio and the image, does it seem more likely than not "
            "that this segment is from a NASCAR race broadcast (not an ad)? Reply 'Yes' or 'No'."
        )
        msg_content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            }
        )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": msg_content}  # pyright: ignore[reportArgumentType]
    ]

    with CLASSIFICATION_TIME.time():
        response = _get_client().chat.completions.create(
            model=app_config.llm_model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.5,
        )

    content = response.choices[0].message.content
    if content is None:
        return False
    return "yes" in content.strip().lower()


def classify_by_prompt(image_data: str, audio_data: str | None = None) -> ClassificationResult:
    """Classify using the full prompt in prompt.txt. Returns the raw LLM reply."""
    msg_content: list = [
        {"type": "text", "text": PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ]
    if audio_data is not None and app_config.enable_llm_audio:
        msg_content.append(
            {
                "type": "input_audio",
                "input_audio": {"data": audio_data, "format": "wav"},
            }
        )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": msg_content}  # pyright: ignore[reportArgumentType]
    ]

    with CLASSIFICATION_TIME.time():
        response = _get_client().chat.completions.create(
            model=app_config.llm_model_name,
            messages=messages,
            max_tokens=500,
            temperature=0.6,
        )

    content = response.choices[0].message.content
    if content is None:
        return ClassificationResult(
            source="llm", type="unknown", reason="empty_response", reply=None
        )
    reply = content.strip().lower()
    return _get_classification_from_response(reply)
