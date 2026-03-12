from flask import Blueprint, current_app, jsonify, render_template, request

from matrix import apply_matrix_settings
from state import state

status_bp = Blueprint("status", __name__)


@status_bp.route("/is_ad/status")
def is_ad_status():
    return jsonify({
        "classification": state.classification,
        "paused": state.paused,
        "auto_switch": state.auto_switch,
        "ad_input_a": current_app.config.get("AD_OUTPUT_SETTING", {}).get("A", "?"),
        "race_input_a": current_app.config.get("RACE_OUTPUT_SETTING", {}).get("A", "?"),
    })


@status_bp.route("/is_ad")
def is_ad():
    return render_template("is_ad.html")


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
