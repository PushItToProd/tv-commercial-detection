from flask import Flask, request, jsonify
from datetime import datetime
from pathlib import Path
import os

app = Flask(__name__)

SAVE_DIR = Path(os.environ.get("SAVE_DIR", "frames"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/save", methods=["POST"])
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
    save_path = SAVE_DIR / filename
    image.save(save_path)

    print(f"Saved: {save_path}  |  page: {request.form.get('page_title', '?')}")
    return jsonify({"saved": filename}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11434, debug=True)
