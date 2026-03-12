import json
from datetime import datetime
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request, send_from_directory

review_bp = Blueprint("review", __name__)


def load_labels() -> dict:
    labels_file = current_app.config["SAVE_DIR"] / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    labels_file = current_app.config["SAVE_DIR"] / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)


@review_bp.route("/save", methods=["POST"])
def save():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    image = request.files["image"]

    # Use the extension's timestamp if provided, otherwise use server time
    raw_ts = request.form.get("timestamp")
    try:
        dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")) if raw_ts else datetime.now()
    except ValueError:
        dt = datetime.now()

    filename = dt.strftime("%Y-%m-%d_%H-%M-%S") + ".png"
    save_dir = current_app.config["SAVE_DIR"]
    save_path = save_dir / filename
    image.save(save_path)

    print(f"Saved: {save_path}  |  page: {request.form.get('page_title', '?')}")
    return jsonify({"saved": filename}), 200


@review_bp.route("/frames/<filename>")
def serve_frame(filename):
    # send_from_directory prevents path traversal
    return send_from_directory(current_app.config["SAVE_DIR"].resolve(), filename)


@review_bp.route("/classify", methods=["POST"])
def handle_classify():
    data = request.get_json()
    if not data or "filename" not in data or "label" not in data:
        return jsonify({"error": "Missing filename or label"}), 400
    label = data["label"]
    if label not in ("ad", "content"):
        return jsonify({"error": "label must be 'ad' or 'content'"}), 400
    filename = data["filename"]
    # Guard against path traversal: filename must be a plain basename ending in .png
    if Path(filename).name != filename or not filename.endswith(".png"):
        return jsonify({"error": "Invalid filename"}), 400
    labels = load_labels()
    labels[filename] = label
    save_labels(labels)
    return jsonify({"classified": filename, "label": label}), 200


@review_bp.route("/review")
def review():
    save_dir = current_app.config["SAVE_DIR"]
    labels = load_labels()
    images = sorted(p.name for p in save_dir.glob("*.png"))
    image_data = [{"filename": f, "label": labels.get(f)} for f in images]
    return render_template("review.html", image_data=image_data)
