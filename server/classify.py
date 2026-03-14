import argparse
import base64
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from openai import OpenAI

import prometheus_client


SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://192.168.1.27:3002")

REFERENCE_LOGO_PATH = Path(__file__).parent / "prompt" / "fs1_logo.png"

CROP_WIDTH_PCT = 0.12
CROP_HEIGHT_PCT = 0.15

PROMPT_FILE = os.environ.get("PROMPT_FILE", Path(__file__).parent / "prompt" / "prompt.txt")
PROMPT = Path(PROMPT_FILE).read_text()

PROMPT_DIR = Path(__file__).parent / "prompt"

MAX_DIMENSION = 800

# Each entry is (image_path, expected_assistant_reply).
EXAMPLES: list[tuple[str, str]] = []


@dataclass
class ClassificationResult:
    type: str  # "ad", "content", or "unknown"
    reason: str
    reply: str | None = None


CLASSIFICATION_TIME = prometheus_client.Histogram(
    "classification_time_seconds",
    "Time spent classifying each image",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 5, 10],
)


def load_examples() -> list[tuple[str, str]]:
    """Load few-shot examples from prompt/ad_frames and prompt/race_frames."""
    examples: list[tuple[str, str]] = []
    for subdir in ("ad_frames", "race_frames"):
        for img_path in sorted((PROMPT_DIR / subdir).glob("*.png")):
            txt_path = img_path.with_suffix(".png.txt")
            if txt_path.exists():
                reply = txt_path.read_text().strip()
                examples.append((str(img_path), reply))
    return examples


def _to_jpeg_b64(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _load_reference_logo() -> str:
    with Image.open(REFERENCE_LOGO_PATH) as img:
        return _to_jpeg_b64(img)


def _crop_upper_right(image_path: str) -> str:
    """Crop the upper-right region (CROP_WIDTH_PCT x CROP_HEIGHT_PCT) and return base64 JPEG."""
    with Image.open(image_path) as img:
        w, h = img.size
        crop_w = round(w * CROP_WIDTH_PCT)
        crop_h = round(h * CROP_HEIGHT_PCT)
        cropped = img.crop((w - crop_w, 0, w, crop_h))
        return _to_jpeg_b64(cropped)


def _resize_image(image_path: str) -> bytes:
    """Resize image so its longest side is at most MAX_DIMENSION, then return JPEG bytes."""
    with Image.open(image_path) as img:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS) # pyright: ignore[reportAttributeAccessIssue]
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()


def _contains_network_logo(image_path: str) -> bool:
    """Send the FS1 reference logo and the cropped upper-right region to the LLM.

    Returns the raw reply from the model ('yes' or 'no').
    """
    reference_data = _load_reference_logo()
    comparison_data = _crop_upper_right(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "The first image is a reference logo. "
                        "The second image is a cropped region from a broadcast screenshot. "
                        "Does the second image contain the same logo as the first image? "
                        "Reply with only 'yes' or 'no'."
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{reference_data}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{comparison_data}"}},
            ],
        }
    ]

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    with CLASSIFICATION_TIME.time():
        response = client.chat.completions.create(
            model="local",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )

    content = response.choices[0].message.content
    if content is None:
        return False
    return "yes" in content.strip().lower()


def _contains_vertical_scoreboard(image_path: str) -> bool:
    """
    Crop the lower-left 20%x80% of the image and ask the LLM if it contains a
    vertical NASCAR/racing scoreboard.

    Returns True if the model replies 'yes'.
    """
    with Image.open(image_path) as img:
        w, h = img.size
        crop_w = round(w * 0.20)
        crop_top = round(h * 0.20)
        cropped = img.crop((0, crop_top, crop_w, h))
        crop_data = _to_jpeg_b64(cropped)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Does this image contain a NASCAR/racing scoreboard? Reply with only 'yes' or 'no'.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_data}"}},
            ],
        }
    ]

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    with CLASSIFICATION_TIME.time():
        response = client.chat.completions.create(
            model="local",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )

    content = response.choices[0].message.content
    if content is None:
        return False
    return "yes" in content.strip().lower()


def _contains_horizontal_scoreboard(image_path: str) -> bool:
    """
    Crop the top 20% of the image and ask the LLM if it contains a horizontal
    NASCAR scoreboard like those used during side-by-side ad breaks.

    Returns the raw reply from the model ('yes' or 'no').
    """
    with Image.open(image_path) as img:
        w, h = img.size
        crop_h = round(h * 0.20)
        cropped = img.crop((0, 0, w, crop_h))
        crop_data = _to_jpeg_b64(cropped)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Does this image contain a NASCAR scoreboard? Reply with only 'yes' or 'no'.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_data}"}},
            ],
        }
    ]

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    with CLASSIFICATION_TIME.time():
        response = client.chat.completions.create(
            model="local",
            messages=messages,
            max_tokens=10,
            temperature=0.0,
        )

    content = response.choices[0].message.content
    if content is None:
        return False
    return "yes" in content.strip().lower()


def _classify_by_prompt(image_path: str) -> ClassificationResult:
    """Classify using the full prompt in prompt.txt. Returns the raw LLM reply."""
    image_data = base64.b64encode(_resize_image(image_path)).decode("utf-8")

    messages = []
    for i, (ex_path, ex_reply) in enumerate(EXAMPLES):
        ex_data = base64.b64encode(_resize_image(ex_path)).decode("utf-8")
        user_content = []
        if i == 0:
            user_content.append({"type": "text", "text": PROMPT})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ex_data}"}})
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": ex_reply})

    final_content = []
    if not EXAMPLES:
        final_content.append({"type": "text", "text": PROMPT})
    final_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}})
    messages.append({"role": "user", "content": final_content})

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
        return ClassificationResult(type="unknown", reason="empty_response")
    reply = content.strip().lower()
    return _get_classification_from_response(reply)


AD_MATCH_REGEX = re.compile(r'\btype=ad\b|\"classification\"\s*:\s*\"ad\"')
RACING_MATCH_REGEX = re.compile(r'\btype=racing\b|\"classification\"\s*:\s*\"racing\"')


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
        return ClassificationResult(type="ad", reason="model-match", reply=reply)
    if RACING_MATCH_REGEX.search(reply):
        return ClassificationResult(type="content", reason="model-match", reply=reply)

    data = _extract_json(reply)
    if data is not None:
        classification = data.get("classification")
        if classification == "racing":
            return ClassificationResult(type="content", reason="model-match", reply=reply)
        if classification == "ad":
            return ClassificationResult(type="ad", reason="model-match", reply=reply)

    return ClassificationResult(type="unknown", reason="model-match", reply=reply)


def classify_image(image_path: str) -> ClassificationResult:
    """Three-pass classification: logo detection, scoreboard detection, then prompt-based fallback."""
    # If it contains the network logo in the upper right, it's racing content.
    logo_reply = _contains_network_logo(image_path)
    if logo_reply:
        return ClassificationResult(type="content", reason="network_logo")

    # If it contains a vertical scoreboard on the left, it's racing content.
    # FIXME: the stupid LLM seems to treat the appearance of a car as a
    # scoreboard sometimes.
    if _contains_vertical_scoreboard(image_path):
        return ClassificationResult(type="content", reason="vertical_scoreboard")

    # If it contains a horizontal scoreboard in the top 20%, it's a side-by-side
    # ad break.
    # FIXME: this actually reduced accuracy compared to just checking the logo.
    if _contains_horizontal_scoreboard(image_path):
        return ClassificationResult(type="ad", reason="side_by_side")

    return _classify_by_prompt(image_path)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--include-reply', action='store_true', help='Whether to include the full LLM reply in the output')
    parser.add_argument('--load-examples', action='store_true', help='(No-op) kept for backwards compatibility')
    return parser


def main():
    args = get_parser().parse_args()

    result = classify_image(args.image_path)
    print(result.type)

    if args.include_reply:
        print(result.reason)


if __name__ == "__main__":
    main()
