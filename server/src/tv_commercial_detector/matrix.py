import asyncio
import json
import logging
import urllib.request

import prometheus_client

from .config import app_config

logger = logging.getLogger(__name__)


SWITCHING_TIME = prometheus_client.Histogram(
    "switching_time_seconds",
    "Time spent switching HDMI matrix after classification",
    buckets=[0.5, 0.75, 1, 1.5, 2, 3, 4, 5],
)

VALID_OUTPUTS = {"A", "B"}


async def apply_matrix_settings(classification: str) -> None:
    """Send HDMI matrix switch commands for the given classification,
    awaiting completion."""
    matrix_url = app_config.matrix_url
    output_settings = app_config.output_settings
    settings: dict = dict(output_settings.get(classification, {}))
    if not settings:
        return

    def _send():
        for output, input_num in settings.items():
            if output == "label":
                continue
            if output not in VALID_OUTPUTS:
                logger.error("invalid output '%s' in settings for classification '%s'", output, classification)
                continue
            payload = json.dumps(
                {
                    "output": output,
                    "input": int(input_num),
                    "quick": True,
                }
            ).encode()
            req = urllib.request.Request(
                f"{matrix_url}/set-output-input",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with SWITCHING_TIME.time():
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        logger.info(
                            "Matrix: output %s → input %s  (%s)",
                            output,
                            input_num,
                            resp.status,
                        )
            except Exception:
                logger.exception(
                    f"Matrix error (output {output} → input {input_num})"
                )

    await asyncio.to_thread(_send)
