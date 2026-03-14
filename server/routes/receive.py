import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from classify import classify_image
from config import app_config
from routes.status import broadcast_status
from routes.trigger_matrix import apply_matrix_settings
from state import last_image_path, recent_frames, state

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/receive")
async def receive(
    image: UploadFile | None = File(default=None),
    is_paused: str = Form(default=""),
    is_seeking: str = Form(default=""),
    page_title: str = Form(default="?"),
    video_title: str = Form(default=""),
    network_name: str = Form(default=""),
    video_offset: str = Form(default=""),
):
    state.paused = is_paused_bool = is_paused.lower() in ("true", "1", "yes")
    state.seeking = is_seeking_bool = is_seeking.lower() in ("true", "1", "yes")
    offset_secs: float | None = float(video_offset) if video_offset else None
    offset_str = f"{offset_secs:.1f}s" if offset_secs is not None else "?"

    if image is None:
        if is_seeking_bool:
            print(f"Seeking (no image)  |  offset: {offset_str}  |  page: {page_title}")
            await broadcast_status()
            return {"classification": state.classification, "paused": False, "seeking": True}
        if is_paused_bool:
            print(f"Paused (no image)  |  offset: {offset_str}  |  page: {page_title}")
            await broadcast_status()
            return {"classification": state.classification, "paused": True}
        raise HTTPException(status_code=400, detail="No image field in request")

    # Preserve the uploaded extension (.jpg or .png) so PIL detects the format correctly
    ext = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"

    frame_bytes = await image.read()

    # Write to a temp file so classify_image (which expects a path) can read it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(frame_bytes)
        tmp_path = tmp.name

    try:
        reply = classify_image(tmp_path)
        result = reply.type
    except Exception:
        logger.exception("Classification error")
        result = "unknown"

    try:
        recent_frames.append((datetime.now().isoformat(), frame_bytes, ext))
        shutil.copy2(tmp_path, last_image_path)
    except Exception:
        logger.exception("Error saving recent frame")
    finally:
        os.unlink(tmp_path)

    classification = state.classification

    # We ignore "unknown" here -- e.g. if we get `content -> unknown ->
    # content`, treat that like two consecutive `content` results and switch to
    # content. Right now any "unknown" in the middle of two identical results
    # will prevent switching, which is not ideal since "unknown" is often just a
    # momentary uncertainty.
    prev = state.last_result
    if result != "unknown":
        state.last_result = result

    apply_new_settings = False

    # Commit only when the same result appears twice in a row and differs from current state
    if (result == prev or not state.enable_debounce) and result != classification and result in ("ad", "content"):
        state.classification = result
        logger.info(f"Classification changed: {classification} → {result}  |  offset: {offset_str}  |  page: {page_title}")
        # Don't actually apply the new settings yet. We want to update the UI
        # first.
        apply_new_settings = state.auto_switch
    else:
        logger.info(f"Received image → classified as: {result}  |  offset: {offset_str}  |  page: {page_title}")

    await broadcast_status()
    if apply_new_settings:
        await apply_matrix_settings(result)

    return {"classification": state.classification, "paused": is_paused_bool}


class ReportWrongRequest(BaseModel):
    correct_label: str


@router.post("/video-state")
async def video_state(
    is_paused: str = Form(default=""),
    is_seeking: str = Form(default=""),
    page_title: str = Form(default="?"),
    page_url: str = Form(default=""),
    video_title: str = Form(default=""),
    network_name: str = Form(default=""),
):
    state.paused = is_paused_bool = is_paused.lower() in ("true", "1", "yes")
    state.seeking = is_seeking_bool = is_seeking.lower() in ("true", "1", "yes")

    status = "paused" if is_paused_bool else "seeking" if is_seeking_bool else "resumed"
    print(f"Video state: {status}  |  page: {page_title}")

    await broadcast_status()
    return {"classification": state.classification, "paused": is_paused_bool, "seeking": is_seeking_bool}


@router.post("/report_wrong")
async def report_wrong(data: ReportWrongRequest):
    correct_label = data.correct_label
    if correct_label not in ("ad", "content"):
        raise HTTPException(status_code=400, detail="correct_label must be 'ad' or 'content'")
    if not recent_frames:
        raise HTTPException(status_code=400, detail="No image available")

    incorrect_dir = app_config.incorrect_dir
    labels_file = incorrect_dir / "incorrect_labels.json"
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)

    saved = []
    for i, (ts, frame_bytes, ext) in enumerate(recent_frames):
        safe_ts = ts.replace(":", "-").replace(".", "-")
        filename = f"{safe_ts}_{i}{ext}"
        dest = incorrect_dir / filename
        dest.write_bytes(frame_bytes)
        labels[filename] = {"correct_label": correct_label, "classified_as": state.classification}
        saved.append(filename)

    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Correction saved: {len(saved)} frame(s) to {incorrect_dir}  |  classified as: {state.classification}, correct: {correct_label}")

    # Update the classification (so we won't immediately switch back if debounce
    # is enabled and we get another wrong classification).
    state.classification = correct_label
    state.last_result = correct_label

    # Immediately switch to the correct output mode.
    state.matrix_switching = True
    await broadcast_status()
    try:
        await apply_matrix_settings(correct_label)
    finally:
        state.matrix_switching = False
        await broadcast_status()

    return {"saved": saved, "correct_label": correct_label}
