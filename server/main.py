import json
import os
from pathlib import Path

from flask import Flask, Response
from prometheus_flask_exporter import PrometheusMetrics

import classify
from routes.receive import receive_bp
from routes.review import review_bp
from routes.status import status_bp


metrics = PrometheusMetrics.for_app_factory()


def create_app() -> Flask:
    app = Flask(__name__)

    metrics.init_app(app)

    # Defaults
    app.config.from_mapping(
        MATRIX_URL="http://localhost:5000",
        AD_OUTPUT_SETTING={},
        RACE_OUTPUT_SETTING={},
        SAVE_DIR=Path("frames"),
        INCORRECT_DIR=Path("incorrect_frames"),
        LOAD_EXAMPLES=False,
    )

    # Load config.json, uppercasing keys to match Flask convention
    config_path = Path(os.environ.get("CONFIG_FILE", "config.json"))
    if config_path.exists():
        with config_path.open() as f:
            app.config.update({k.upper(): v for k, v in json.load(f).items()})

    # Environment variable overrides: DETECTOR_MATRIX_URL, DETECTOR_SAVE_DIR, etc.
    app.config.from_prefixed_env("DETECTOR")

    # Ensure path types and create directories
    app.config["SAVE_DIR"] = Path(app.config["SAVE_DIR"])
    app.config["INCORRECT_DIR"] = Path(app.config["INCORRECT_DIR"])
    app.config["SAVE_DIR"].mkdir(parents=True, exist_ok=True)
    app.config["INCORRECT_DIR"].mkdir(parents=True, exist_ok=True)

    if app.config["LOAD_EXAMPLES"]:
        classify.EXAMPLES = classify.load_examples()

    app.register_blueprint(receive_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(status_bp)

    @app.route('/metrics')
    def get_metrics():
        metrics_response, content_type = metrics.generate_metrics()
        return Response(metrics_response, mimetype=content_type)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=11434, debug=True)
