import base64
import sys
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


def classify_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    client = OpenAI(base_url=f"{SERVER_URL}/v1", api_key="none")

    response = client.chat.completions.create(
        model="local",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    },
                ],
            }
        ],
        max_tokens=500,
    )

    reply = response.choices[0].message.content.strip().lower()

    if "type=ad" in reply:
        return "ad"
    elif "type=racing" in reply:
        return "content"
    else:
        print(f"Warning: Could not determine classification from model response: '{reply}'")
        return "unknown"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <image.png>")
        sys.exit(1)

    result = classify_image(sys.argv[1])
    print(result)
