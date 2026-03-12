import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request

from classify import classify_image
from matrix import apply_matrix_settings
from state import last_image_path, recent_frames, state

receive_bp = Blueprint("receive", __name__)


@receive_bp.route("/receive", methods=["POST"])
def receive():
    is_paused = request.form.get("is_paused", "").lower() in ("true", "1", "yes")
    state.paused = is_paused

    if "image" not in request.files:
        if is_paused:
            print(f"Paused (no image)  |  page: {request.form.get('page_title', '?')}")
            return jsonify({"classification": state.classification, "paused": True}), 200
        return jsonify({"error": "No image field in request"}), 400

    image = request.files["image"]

    # Preserve the uploaded extension (.jpg or .png) so PIL detects the format correctly
    ext = Path(image.filename).suffix.lower() if image.filename else ".jpg"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"

    # Write to a temp file so classify_image (which expects a path) can read it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = classify_image(tmp_path)
    except Exception as e:
        current_app.logger.exception(f"Classification error")
        result = "unknown"

    try:
        with open(tmp_path, "rb") as f:
            frame_bytes = f.read()
        recent_frames.append((datetime.now().isoformat(), frame_bytes, ext))
        shutil.copy2(tmp_path, last_image_path)
    except Exception as e:
        current_app.logger.exception("Error saving recent frame")
    finally:
        os.unlink(tmp_path)

    committed = state.classification
    pending = state.last_result
    state.last_result = result
    # Commit only when the same result appears twice in a row and differs from current state
    if result == pending and result != committed and result in ("ad", "content"):
        state.classification = result
        current_app.logger.info(f"Classification changed: {committed} → {result}  |  page: {request.form.get('page_title', '?')}")
        if state.auto_switch:
            apply_matrix_settings(result)
    else:
        current_app.logger.info(f"Received image → classified as: {result}  |  page: {request.form.get('page_title', '?')}")

    return jsonify({"classification": state.classification, "paused": is_paused}), 200


@receive_bp.route("/report_wrong", methods=["POST"])
def report_wrong():
    data = request.get_json()
    if not data or "correct_label" not in data:
        return jsonify({"error": "Missing correct_label"}), 400
    correct_label = data["correct_label"]
    if correct_label not in ("ad", "content"):
        return jsonify({"error": "correct_label must be 'ad' or 'content'"}), 400
    if not recent_frames:
        return jsonify({"error": "No image available"}), 400

    incorrect_dir = current_app.config["INCORRECT_DIR"]
    labels_file = incorrect_dir / "labels.json"
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
    return jsonify({"saved": saved, "correct_label": correct_label}), 200
