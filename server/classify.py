import base64
import io
import sys
from PIL import Image
from openai import OpenAI

SERVER_URL = "http://192.168.1.27:3002"

PROMPT = (
    "You are analyzing a screenshot from a TV broadcast. "
    "Determine whether the image shows a TV commercial/advertisement, "
    "or actual race content (cars, track, drivers, pit lane, etc.). "
    "In 100 words or less, describe what you see in the image. "
    "Then, based on that description, end your message with either 'type=ad' or "
    "'type=racing'."
)


MAX_DIMENSION = 1024

# Each entry is (image_path, expected_assistant_reply).
# The reply should be a short description followed by the label, e.g.:
#   "A car dealership ad with a red sedan. type=ad"
EXAMPLES: list[tuple[str, str]] = []


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
    if reply.endswith("type=ad"):
        return "ad"
    elif reply.endswith("type=racing"):
        return "content"
    else:
        return "unknown"


def classify_image(image_path: str) -> str:
    reply = _classify_image(image_path)
    return get_classification_from_response(reply)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image.png>")
        sys.exit(1)

    result = classify_image(sys.argv[1])
    print(result)
