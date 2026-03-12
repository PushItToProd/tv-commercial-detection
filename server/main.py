import os

import classify
from flask import Flask
from routes.receive import receive_bp
from routes.review import review_bp
from routes.status import status_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.register_blueprint(receive_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(status_bp)
    return app


if __name__ == "__main__":
    if os.environ.get("LOAD_EXAMPLES", "").lower() in ("1", "true", "yes"):
        classify.EXAMPLES = classify.load_examples()
    app = create_app()
    app.run(host="0.0.0.0", port=11434, debug=True)
