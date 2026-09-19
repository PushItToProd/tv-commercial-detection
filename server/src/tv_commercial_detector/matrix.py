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
    """Send one routing batch and wait for the service to drain its replies.

    The classification supplies absolute routes, so no observation version is
    needed. An acknowledgement does not prove the device applied the routes.
    """
    matrix_url = app_config.matrix_url
    settings = dict(app_config.output_settings.get(classification, {}))
    routes = {}
    for output, input_num in settings.items():
        if output == "label":
            continue
        if output not in VALID_OUTPUTS:
            logger.error(
                "invalid output '%s' in settings for classification '%s'",
                output,
                classification,
            )
            continue
        try:
            # Configured inputs may be strings, but never silently truncate
            # fractional values or treat booleans as input numbers.
            if isinstance(input_num, bool) or not isinstance(input_num, (int, str)):
                raise ValueError("input must be an integer")
            inp = int(input_num)
            if inp not in (1, 2, 3, 4):
                raise ValueError("input must be 1-4")
        except ValueError, TypeError:
            logger.error("invalid input %r for output %s", input_num, output)
            return  # do not send a partially validated batch
        routes[output] = inp
    if not routes:
        return

    def _send():
        req = urllib.request.Request(
            f"{matrix_url.rstrip('/')}/apply",
            data=json.dumps(routes).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with SWITCHING_TIME.time():
                with urllib.request.urlopen(req, timeout=5) as resp:
                    logger.info(
                        "Matrix: routing %s acknowledged (%s)", routes, resp.status
                    )
        except Exception:
            # A timeout may follow a partial or complete switch; do not replay.
            logger.exception("Matrix error (routing %s)", routes)

    await asyncio.to_thread(_send)
