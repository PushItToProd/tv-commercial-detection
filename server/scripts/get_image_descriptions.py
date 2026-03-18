import json
import sys
import time

from PIL import Image
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from tv_commercial_detector import classify
import check_classification


def get_image_description(image_path: str, server_url: str = classify.SERVER_URL) -> str | None:
    """
    Just ask the LLM to describe the image, without giving it any specific
    instructions about what to look for. This is intended to help us understand
    what features the model is picking up on when it classifies an image as an
    ad or content.
    """
    with Image.open(image_path) as img:
        crop_data = classify._to_jpeg_b64(img)

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in 100 words or less.",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_data}"}},
            ],
        }
    ]

    client = OpenAI(base_url=f"{server_url}/v1", api_key="none")

    response = client.chat.completions.create(
        model="local",
        messages=messages,
        max_tokens=500,
        temperature=0.6,
    )

    content = response.choices[0].message.content
    if content is None:
        return None
    return content.strip().lower()


def main():
    image_files = check_classification.get_images()

    with check_classification.LABELS_PATH.open() as f:
        labels = json.load(f)

    for f in image_files:
        sys.stdout.flush()

        start_time = time.perf_counter()
        description = get_image_description(f.as_posix())
        elapsed_time = time.perf_counter() - start_time

        print(json.dumps({
            "file": f.name,
            "description": description,
            "elapsed": round(elapsed_time, 2),
            "correct_classification": labels.get(f.name),
        }))


if __name__ == '__main__':
    main()

