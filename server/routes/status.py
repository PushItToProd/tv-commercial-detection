from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from matrix import apply_matrix_settings
from state import last_image_path, state

status_bp = Blueprint("status", __name__)


@status_bp.route("/is_ad/status")
def is_ad_status():
    output_settings = current_app.config.get('OUTPUT_SETTINGS', {})
    ad_view_label = output_settings.get('ad', {}).get('label', 'Ad view')
    race_view_label = output_settings.get('content', {}).get('label', 'Race view')

    return jsonify({
        "classification": state.classification,
        "paused": state.paused,
        "pending": state.is_pending_change(),
        "auto_switch": state.auto_switch,
        "ad_view_label": ad_view_label,
        "race_view_label": race_view_label,
    })


@status_bp.route("/is_ad")
def is_ad():
    return render_template("is_ad.html")


@status_bp.route("/is_ad/last_frame")
def last_frame():
    if not last_image_path.exists():
        return ("", 204)
    return send_file(last_image_path, mimetype="image/jpeg")


@status_bp.route("/auto_switch", methods=["POST"])
def set_auto_switch():
    data = request.get_json()
    if not data or "enabled" not in data:
        return jsonify({"error": "Missing enabled"}), 400
    state.auto_switch = bool(data["enabled"])
    return jsonify({"auto_switch": state.auto_switch}), 200


@status_bp.route("/trigger_matrix", methods=["POST"])
def trigger_matrix():
    data = request.get_json()
    if not data or "classification" not in data:
        return jsonify({"error": "Missing classification"}), 400
    classification = data["classification"]
    if classification not in ("ad", "content"):
        return jsonify({"error": "classification must be 'ad' or 'content'"}), 400
    apply_matrix_settings(classification)
    return jsonify({"triggered": classification}), 200
