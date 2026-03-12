import json
import logging
import threading
import urllib.request

from flask import current_app

import prometheus_client


logger = logging.getLogger(__name__)


SWITCHING_TIME = prometheus_client.Histogram(
    "switching_time_seconds",
    "Time spent switching HDMI matrix after classification",
    buckets=[0.5, 0.75, 1, 1.5, 2, 3, 4, 5],
)


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
                with SWITCHING_TIME.time():
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        logger.info(f"Matrix: output {output} → input {input_num}  ({resp.status})")
            except Exception as e:
                logger.exception(f"Matrix error (output {output} → input {input_num})")

    threading.Thread(target=_send, daemon=True).start()
