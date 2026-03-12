import json
import os
import threading
import urllib.request
from pathlib import Path

CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", "config.json"))


def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            config = json.load(f)
        if os.environ.get("HDMI_MATRIX_URL"):
            config["matrix_url"] = os.environ["HDMI_MATRIX_URL"]
        return config
    return {}


def apply_matrix_settings(classification: str) -> None:
    """Send HDMI matrix switch commands for the given classification in a background thread."""
    config = load_config()
    matrix_url = config.get("matrix_url", "http://localhost:5000")
    key = "ad_output_setting" if classification == "ad" else "race_output_setting"
    settings: dict = config.get(key, {})
    if not settings:
        return

    def _send():
        for output, input_num in settings.items():
            payload = json.dumps({"output": output, "input": int(input_num)}).encode()
            req = urllib.request.Request(
                f"{matrix_url}/set-output-input",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    print(f"Matrix: output {output} → input {input_num}  ({resp.status})")
            except Exception as e:
                print(f"Matrix error (output {output} → input {input_num}): {e}")

    threading.Thread(target=_send, daemon=True).start()
