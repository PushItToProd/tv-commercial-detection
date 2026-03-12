import json
import threading
import urllib.request

from flask import current_app


def apply_matrix_settings(classification: str) -> None:
    """Send HDMI matrix switch commands for the given classification in a background thread."""
    # Capture config values before spawning thread — current_app is not available off-context
    matrix_url = current_app.config["MATRIX_URL"]
    key = "AD_OUTPUT_SETTING" if classification == "ad" else "RACE_OUTPUT_SETTING"
    settings: dict = dict(current_app.config.get(key, {}))
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
