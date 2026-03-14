import argparse
import base64
import io
import json
import os
import re
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


def _classify_by_logo(image_path: str) -> str:
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
        return "no"
    return content.strip().lower()


def _classify_by_prompt(image_path: str) -> str:
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
        return "unknown"
    return content.strip().lower()


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


def get_classification_from_response(reply: str) -> str:
    """Interpret a raw LLM reply from either pass.

    A logo-detection 'yes' maps to 'content'. A prompt-based reply is parsed
    via JSON then regex fallback.
    """
    if reply.startswith("yes"):
        return "content"

    data = _extract_json(reply)
    if data is not None:
        classification = data.get("classification")
        if classification == "racing":
            return "content"
        if classification == "ad":
            return "ad"

    if AD_MATCH_REGEX.search(reply):
        return "ad"
    if RACING_MATCH_REGEX.search(reply):
        return "content"
    return "unknown"


def _classify_image(image_path: str) -> str:
    """Two-pass classification: logo detection first, prompt-based fallback.

    Returns the raw LLM reply suitable for get_classification_from_response.
    """
    logo_reply = _classify_by_logo(image_path)
    if logo_reply.startswith("yes"):
        return logo_reply
    return _classify_by_prompt(image_path)


def classify_image(image_path: str) -> str:
    reply = _classify_image(image_path)
    return get_classification_from_response(reply)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--include-reply', action='store_true', help='Whether to include the full LLM reply in the output')
    parser.add_argument('--load-examples', action='store_true', help='(No-op) kept for backwards compatibility')
    return parser


def main():
    args = get_parser().parse_args()

    if args.include_reply:
        reply = _classify_image(args.image_path)
        print(reply)
        return

    result = classify_image(args.image_path)
    print(result)


if __name__ == "__main__":
    main()
