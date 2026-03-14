import argparse
import base64
import io
import os
from pathlib import Path

from PIL import Image
from openai import OpenAI

import prometheus_client


SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://192.168.1.27:3002")

REFERENCE_LOGO_PATH = Path(__file__).parent / "prompt" / "fs1_logo.png"

CROP_WIDTH_PCT = 0.12
CROP_HEIGHT_PCT = 0.15

# Kept for backwards compatibility with main.py and check_classification.py.
EXAMPLES: list[tuple[str, str]] = []


CLASSIFICATION_TIME = prometheus_client.Histogram(
    "classification_time_seconds",
    "Time spent classifying each image",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 1.75, 2, 5, 10],
)


def load_examples() -> list[tuple[str, str]]:
    """No-op kept for backwards compatibility."""
    return []


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


def _classify_image(image_path: str) -> str:
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


def get_classification_from_response(reply: str) -> str:
    """Interpret LLM reply: 'yes' -> 'content' (racing), anything else -> 'ad'."""
    if reply.startswith("yes"):
        return "content"
    return "ad"


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
