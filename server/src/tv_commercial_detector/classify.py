import argparse
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import prometheus_client
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image

from .classification import logo_match, rectangle_match

SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://192.168.1.27:3002")

PROMPT_DIR = Path(__file__).parent / "prompt"
PROMPT_FILE = os.environ.get("PROMPT_FILE", PROMPT_DIR / "prompt.txt")
PROMPT = Path(PROMPT_FILE).read_text()

MAX_DIMENSION = 800


@dataclass
class ClassificationResult:
    source: str
    type: str  # "ad", "content", or "unknown"
    reason: str
    reply: str | None


CLASSIFICATION_TIME = prometheus_client.Histogram(
    "classification_time_seconds",
    "Time spent classifying each image",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 5, 10],
)


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


def _report_racing_related(image_data: str) -> bool:
    """
    Just ask the LLM "Does this image contain anything related to NASCAR racing?
    Answer 'Yes' or 'No'".
    """
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
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
            ],
        }
    ]

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    with CLASSIFICATION_TIME.time():
        response = client.chat.completions.create(
            model="local",
            messages=messages,
            max_tokens=10,
            temperature=0.5,
        )

    content = response.choices[0].message.content
    if content is None:
        return False
    return "yes" in content.strip().lower()


def _classify_by_prompt(image_data: str) -> ClassificationResult:
    """Classify using the full prompt in prompt.txt. Returns the raw LLM reply."""
    # image_data = base64.b64encode(_resize_image(image_path)).decode("utf-8")

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        },
    ]

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    with CLASSIFICATION_TIME.time():
        response = client.chat.completions.create(
            model="local",
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


AD_MATCH_REGEX = re.compile(r"\btype=ad\b|\"classification\"\s*:\s*\"ad\"")
RACING_MATCH_REGEX = re.compile(r"\btype=racing\b|\"classification\"\s*:\s*\"racing\"")


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
    """Interpret a raw LLM reply from either pass.

    A logo-detection 'yes' maps to 'content'. A prompt-based reply is parsed
    via JSON then regex fallback.
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


def classify_image(image_path: str) -> ClassificationResult:
    """Three-pass classification: logo detection, scoreboard detection,
    then prompt-based fallback."""
    # FIXME: don't pass the image as a path to every function here. Load it once
    # and pass the image object or bytes to each function.

    cv_img = cv2.imread(image_path)

    # Network logo in the upper right -> not an ad break
    if logo_match.has_network_logo(cv_img):
        return ClassificationResult(
            source="opencv", type="content", reason="network_logo", reply="(opencv)"
        )

    # Side-by-side logo in the upper left -> very likely an ad break
    if logo_match.has_side_by_side_logo(cv_img):
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="side_by_side",
            reply="side-by-side logo match (opencv)",
        )

    # check for bounding boxes used during side-by-side ad breaks
    matched_rect = rectangle_match.image_has_known_ad_rectangle(cv_img)
    if matched_rect is not None:
        return ClassificationResult(
            source="opencv",
            type="ad",
            reason="matched_rectangle",
            reply=f"{matched_rect} (opencv)",
        )

    image_data = load_image_b64(image_path)

    racing_related = _report_racing_related(image_data)
    if not racing_related:
        return ClassificationResult(
            source="llm",
            type="ad",
            reason="model_quick_reject",
            reply="No NASCAR-related content detected",
        )

    # # XXX: uncomment to test without using the slower model prompt
    # return ClassificationResult(type="content", reason="assume_content", reply="")

    # This seems to have a negligible effect on accuracy and is pretty slow,
    # so skipping for now:
    # racing_related_pct = _report_racing_related_percentage(image_data)
    # if racing_related_pct <= 10:
    #     return ClassificationResult(
    #         type="ad", reason="low_racing_related_percentage",
    #         reply=f"{racing_related_pct}% racing-related"
    #     )

    return _classify_by_prompt(image_data)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", type=str, help="Path to the image to classify")
    parser.add_argument(
        "--include-reply",
        action="store_true",
        help="Whether to include the full LLM reply in the output",
    )
    return parser


def main():
    args = get_parser().parse_args()

    result = classify_image(args.image_path)
    print(result.type)

    if args.include_reply:
        print(result.reason)


if __name__ == "__main__":
    main()
