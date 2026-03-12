import argparse
import base64
import io
import sys
from pathlib import Path
from PIL import Image
from openai import OpenAI

SERVER_URL = "http://192.168.1.27:3002"

PROMPT = (
    "You are analyzing a screenshot from a Fox Sports NASCAR Cup Series TV broadcast. "
    "Determine whether the broadcast has cut to a commercial break (type=ad), "
    "or whether it is showing actual race content (type=racing).\n\n"
    "RACE BROADCAST — strong indicators, classify as type=racing:\n"
    "- A vertical scoring strip/leaderboard along the LEFT edge showing driver positions, lap data, or car numbers (may include a sponsor logo in the top left)\n"
    "- Race cars on a track (wide shot, aerial view, or in-car cockpit/dashboard view)\n"
    "- Pit lane or pit road activity\n"
    "- Drivers, crew members, or on-air commentators at the race venue or broadcast booth\n"
    "- A studio segment with commentators standing in front of a branded Fox backdrop or holding microphones with the Fox logo\n"
    "- Pre/post-race analysis: track maps, statistics, replays\n"
    "- Sponsor logos in the CORNER of the screen, particularly the upper-left and lower-right (e.g. Wendy's bug, Sonic logo) — "
    "these are normal in-broadcast overlays, NOT indicators of an ad\n"
    "- Sponsor banners overlaid on top of racing footage\n"
    "- A brief network promo or lower-third for another event shown over the race broadcast\n\n"
    "AD BREAK — strong indicators, classify as type=ad:\n"
    "- 'Fox side-by-side': race footage shrunk to to the LEFT side with a horizontal scoring leaderboard at the TOP of the screen while a product ad fills the right half and a sponsor logo is displayed in the lower left\n"
    "- A product (food, vehicle, consumer goods) as the primary visual subject\n"
    "- Lifestyle or non-racing scenarios (people in a home, restaurant, or outdoor setting)\n"
    "- A brand logo or slogan dominating most of the screen\n"
    # TODO: have to adjust this for other series
    "- Any type of race cars other than NASCAR Cup Series cars (e.g. open-wheel cars, Formula 1, IndyCar, NASCAR Trucks, sports cars, motorcycles)\n"
    "- Highly cinematic shots of race cars or racing scenes that are more typical of commercials than live sports broadcasts (e.g. extreme closeups and camera angles that aren't typical of live race broadcasts)\n"
    "- Racing-themed imagery (cars, drivers) used to sell a product rather than show live action\n\n"
    "In 100 words or less, describe what you see in the image. "
    "Then end your message with either 'type=ad' or 'type=racing'."
)


MAX_DIMENSION = 1024

# Each entry is (image_path, expected_assistant_reply).
# The reply should be a short description followed by the label, e.g.:
#   "the image shows a tv broadcast of a nascar cup series race event. type=racing"
EXAMPLES: list[tuple[str, str]] = []

PROMPT_DIR = Path(__file__).parent / "prompt"


def load_examples() -> list[tuple[str, str]]:
    """Load few-shot examples from prompt/ad_frames and prompt/race_frames.

    Each subdirectory contains .png files and paired .png.txt files whose
    content is the expected assistant reply for that image.
    """
    examples: list[tuple[str, str]] = []
    for subdir in ("ad_frames", "race_frames"):
        # TODO: make sure we handle the case where the prompt directory doesn't exist
        for img_path in sorted((PROMPT_DIR / subdir).glob("*.png")):
            txt_path = img_path.with_suffix(".png.txt")
            if txt_path.exists():
                reply = txt_path.read_text().strip()
                examples.append((str(img_path), reply))
    return examples


def _resize_image(image_path: str) -> bytes:
    """Resize image so its longest side is at most MAX_DIMENSION, then return JPEG bytes."""
    with Image.open(image_path) as img:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()


def encode_image(image_path):
    image_bytes = _resize_image(image_path)
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    return image_data


def _classify_image(image_path: str) -> str:
    image_data = encode_image(image_path)

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

    response = client.chat.completions.create(
        model="local",
        messages=messages,
        max_tokens=500,
        temperature=0.6,
    )

    content = response.choices[0].message.content
    if content is None:
        return "unknown"
    reply = content.strip().lower()
    return reply


def get_classification_from_response(reply: str) -> str:
    if reply.endswith("type=ad") or "type=ad" in reply:
        return "ad"
    elif reply.endswith("type=racing") or "type=racing" in reply:
        return "content"
    else:
        return "unknown"


def classify_image(image_path: str) -> str:
    reply = _classify_image(image_path)
    return get_classification_from_response(reply)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='Path to the image to classify')
    parser.add_argument('--include-reply', action='store_true', help='Whether to include the full assistant reply in the output')
    parser.add_argument('--load-examples', action='store_true', help='Whether to load few-shot examples from the prompt directory')
    return parser


def main():
    args = get_parser().parse_args()

    if args.load_examples:
        global EXAMPLES
        EXAMPLES = load_examples()

    if args.include_reply:
        reply = _classify_image(args.image_path)
        # print("Assistant reply:")
        print(reply)
        classification = get_classification_from_response(reply)
        # print(f"Classification: {classification}")
        _ = classification  # avoid unused variable warning
        return

    result = classify_image(args.image_path)
    print(result)


if __name__ == "__main__":
    main()
