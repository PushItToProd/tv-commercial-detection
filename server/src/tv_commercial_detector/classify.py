import argparse
import importlib
from pathlib import Path

from .classification.result import ClassificationResult
from .config import app_config


def classify_image(image_path: str) -> ClassificationResult:
    """Dispatch to the active classifier profile."""
    module = importlib.import_module(
        f".classifiers.{app_config.classifier_profile}",
        package=__package__,
    )
    return module.classify_image(image_path)


def list_profiles() -> list[str]:
    """Return sorted list of available classifier profile names."""
    classifiers_dir = Path(__file__).parent / "classifiers"
    return sorted(
        p.stem
        for p in classifiers_dir.glob("*.py")
        if p.stem != "__init__"
    )


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
