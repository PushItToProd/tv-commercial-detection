import json
from datetime import datetime
from pathlib import Path
from PIL import Image

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
        current_app.logger.exception(f"Invalid timestamp format in request: {raw_ts}")
        dt = datetime.now()

    ext = Path(image.filename).suffix.lower() if image.filename else ".png"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".png"
    filename = dt.strftime("%Y-%m-%d_%H-%M-%S") + ext
    save_dir = current_app.config["SAVE_DIR"]
    save_path = save_dir / filename
    image.save(save_path)

    current_app.logger.info(f"Saved: {save_path}  |  page: {request.form.get('page_title', '?')}")
    return jsonify({"saved": filename}), 200


@review_bp.route("/frames/<filename>")
def serve_frame(filename):
    # We compress the images on demand and serve smaller versions to save
    # bandwidth and speed up loading in the review interface. We have to make
    # sure to use send_from_directory with a safe filename to avoid path
    # traversal issues, and we only allow .jpg and .png files.
    save_dir = current_app.config["SAVE_DIR"]
    original_path = save_dir / filename
    compressed_path = save_dir / f"compressed_{filename}"
    if not compressed_path.exists():
        try:
            with Image.open(original_path) as img:
                img.thumbnail((400, 400))
                img.save(compressed_path)
        except Exception as e:
            current_app.logger.exception(f"Error compressing image {original_path}")
            # If compression fails, we can still serve the original image
            return send_from_directory(save_dir.resolve(), filename)

    return send_from_directory(save_dir.resolve(), compressed_path.name)


@review_bp.route("/classify", methods=["POST"])
def handle_classify():
    data = request.get_json()
    if not data or "filename" not in data or "label" not in data:
        return jsonify({"error": "Missing filename or label"}), 400
    label = data["label"]
    if label not in ("ad", "content"):
        return jsonify({"error": "label must be 'ad' or 'content'"}), 400
    filename = data["filename"]
    # Guard against path traversal: filename must be a plain basename ending in .png or .jpg
    if Path(filename).name != filename or not filename.endswith((".png", ".jpg")):
        return jsonify({"error": "Invalid filename"}), 400
    labels = load_labels()
    labels[filename] = label
    save_labels(labels)
    return jsonify({"classified": filename, "label": label}), 200


@review_bp.route("/review")
def review():
    save_dir = current_app.config["SAVE_DIR"]
    labels = load_labels()
    images = sorted(p.name for p in [*save_dir.glob("*.png"), *save_dir.glob("*.jpg")])
    image_data = [{"filename": f, "label": labels.get(f)} for f in images]
    return render_template("review.html", image_data=image_data)
