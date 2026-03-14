import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

import classify
from config import app_config
from frame_saver import periodic_frame_saver
from metrics import instrumentator
from routes.receive import router as receive_router
from routes.review import router as review_router
from routes.status import router as status_router
from routes.trigger_matrix import router as matrix_router
from state import state

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config.json
    config_path = Path(os.environ.get("CONFIG_FILE", "config.json"))
    if config_path.exists():
        with config_path.open() as f:
            for k, v in json.load(f).items():
                if hasattr(app_config, k.lower()):
                    setattr(app_config, k.lower(), v)

    # Environment variable overrides: DETECTOR_MATRIX_URL, DETECTOR_SAVE_DIR, etc.
    env_map = {
        "DETECTOR_MATRIX_URL": "matrix_url",
        "DETECTOR_SAVE_DIR": "save_dir",
        "DETECTOR_LOAD_EXAMPLES": "load_examples",
        "DETECTOR_ENABLE_DEBOUNCE": "enable_debounce",
    }
    for env_key, attr in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            setattr(app_config, attr, val)

    # Ensure path type and create directory
    app_config.save_dir = Path(app_config.save_dir)
    app_config.save_dir.mkdir(parents=True, exist_ok=True)

    if app_config.load_examples:
        classify.EXAMPLES = classify.load_examples()

    state.enable_debounce = app_config.enable_debounce

    saver_task = asyncio.create_task(periodic_frame_saver())
    yield
    saver_task.cancel()
    try:
        await saver_task
    except asyncio.CancelledError:
        pass


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    instrumentator.instrument(app).expose(app)

    app.include_router(receive_router)
    app.include_router(review_router)
    app.include_router(status_router)
    app.include_router(matrix_router)

    return app


if __name__ == "__main__":
    import uvicorn
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=11434)
