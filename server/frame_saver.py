import asyncio
import json
import logging
from datetime import datetime

from config import app_config
from state import FrameEntry, recent_frames, state

logger = logging.getLogger(__name__)

CLASSIFICATIONS_FILE = "classifications.jsonl"


def save_frames_batch(frames: list[FrameEntry], save_reason: str, extra: dict | None = None) -> list[str]:
    """Save a batch of FrameEntry items to save_dir and append metadata to classifications.jsonl.

    Returns a list of saved filenames.
    """
    save_dir = app_config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    classifications_file = save_dir / CLASSIFICATIONS_FILE

    saved: list[str] = []
    with classifications_file.open("a") as f:
        for i, entry in enumerate(frames):
            safe_ts = entry.timestamp.replace(":", "-").replace(".", "-")
            filename = f"{safe_ts}_{i}{entry.ext}"
            dest = save_dir / filename
            try:
                dest.write_bytes(entry.frame_bytes)
            except Exception:
                logger.exception(f"Error saving frame {filename}")
                continue
            saved.append(filename)

            record: dict = {
                "filename": filename,
                "timestamp": entry.timestamp,
                "save_reason": save_reason,
                "page_title": entry.page_title,
                "video_title": entry.video_title,
                "network_name": entry.network_name,
                "video_offset": entry.video_offset,
                "state_classification": entry.state_classification,
            }
            if entry.result is not None:
                record["classification"] = entry.result.type
                record["classification_reason"] = entry.result.reason
                record["model_reply"] = entry.result.reply
            if extra:
                record.update(extra)
            f.write(json.dumps(record) + "\n")

    return saved


async def periodic_frame_saver() -> None:
    """Background task: save recent_frames once per minute for posterity."""
    while True:
        await asyncio.sleep(60)
        if not recent_frames:
            continue
        cutoff = state.last_periodic_save
        frames_to_save = [
            f for f in recent_frames
            if cutoff is None or datetime.fromisoformat(f.timestamp) > cutoff
        ]
        if frames_to_save:
            saved = save_frames_batch(frames_to_save, "periodic")
            state.last_periodic_save = datetime.now()
            if saved:
                logger.info(f"Periodic save: {len(saved)} frame(s)")
