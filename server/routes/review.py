import json
from datetime import datetime
from pathlib import Path
import logging
from PIL import Image

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config import app_config

logger = logging.getLogger(__name__)
router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def load_labels() -> dict:
    labels_file = app_config.save_dir / "labels.json"
    if labels_file.exists():
        with open(labels_file) as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    labels_file = app_config.save_dir / "labels.json"
    with open(labels_file, "w") as f:
        json.dump(labels, f, indent=2)


# must be kept in sync with featFields in review.html
VALID_NETWORK_LOGOS = frozenset({"Fox", "FS1", "FS2", "NBC", "CW", "USA", "other", "none"})
VALID_LOGO_POSITIONS = frozenset({"upper_left", "upper_right", "lower_left", "lower_right", "not_visible", "unknown"})
VALID_SCOREBOARD_POSITIONS = frozenset({"top", "bottom", "left", "upper_left", "right", "none", "unknown"})


def load_features() -> dict:
    features_file = app_config.save_dir / "features.jsonl"
    result: dict = {}
    if features_file.exists():
        with open(features_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    result[record["filename"]] = record
    return result


def save_features(features: dict) -> None:
    features_file = app_config.save_dir / "features.jsonl"
    with open(features_file, "w") as f:
        for record in features.values():
            f.write(json.dumps(record) + "\n")


@router.post("/save")
async def save(
    image: UploadFile = File(...),
    timestamp: str = Form(default=""),
    page_title: str = Form(default="?"),
    video_title: str = Form(default=""),
    network_name: str = Form(default=""),
):
    # Use the extension's timestamp if provided, otherwise use server time
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if timestamp else datetime.now()
    except ValueError:
        dt = datetime.now()

    ext = Path(image.filename).suffix.lower() if image.filename else ".png"
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".png"
    filename = dt.strftime("%Y-%m-%d_%H-%M-%S") + ext
    save_dir = app_config.save_dir
    save_path = save_dir / filename
    save_path.write_bytes(await image.read())

    logger.info(f"Saved: {save_path}  |  page: {page_title}")
    return {"saved": filename}


@router.get("/frames/{filename}")
def serve_frame(filename: str):
    # Guard against path traversal: filename must be a plain basename ending in .jpg or .png
    if Path(filename).name != filename or not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid filename")

    save_dir = app_config.save_dir

    if filename.startswith("compressed_"):
        raise HTTPException(status_code=404, detail="File not found")

    original_path = save_dir / filename
    compressed_path = save_dir / f"compressed_{filename}"
    if not compressed_path.exists():
        try:
            with Image.open(original_path) as img:
                img.thumbnail((400, 400))
                img.save(compressed_path)
        except Exception:
            logger.exception(f"Error compressing image {original_path}")
            if not original_path.exists():
                raise HTTPException(status_code=404, detail="File not found")
            return FileResponse(original_path)

    return FileResponse(compressed_path)


@router.get("/frames/full/{filename}")
def serve_frame_full(filename: str):
    """Serve the original (uncompressed) frame, used by the step-through review view."""
    if Path(filename).name != filename or not filename.endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if filename.startswith("compressed_"):
        raise HTTPException(status_code=404, detail="File not found")
    path = app_config.save_dir / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


class ClassifyRequest(BaseModel):
    filename: str
    label: str


class FeaturesRequest(BaseModel):
    filename: str
    network_logo: str | None = None
    logo_position: str | None = None
    scoreboard_position: str | None = None


@router.post("/classify")
def handle_classify(data: ClassifyRequest):
    label = data.label
    if label not in ("ad", "content", "ignore"):
        raise HTTPException(status_code=400, detail="label must be 'ad', 'content', or 'ignore'")

    filename = data.filename
    # Guard against path traversal: filename must be a plain basename ending in .png or .jpg
    if Path(filename).name != filename or not filename.endswith((".png", ".jpg")):
        raise HTTPException(status_code=400, detail="Invalid filename")
    labels = load_labels()
    labels[filename] = label
    save_labels(labels)
    return {"classified": filename, "label": label}


@router.post("/features")
def handle_features(data: FeaturesRequest):
    if data.network_logo is not None and data.network_logo not in VALID_NETWORK_LOGOS:
        raise HTTPException(status_code=400, detail="Invalid network_logo value")
    if data.logo_position is not None and data.logo_position not in VALID_LOGO_POSITIONS:
        raise HTTPException(status_code=400, detail="Invalid logo_position value")
    if data.scoreboard_position is not None and data.scoreboard_position not in VALID_SCOREBOARD_POSITIONS:
        raise HTTPException(status_code=400, detail="Invalid scoreboard_position value")

    filename = data.filename
    if Path(filename).name != filename or not filename.endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid filename")

    features = load_features()
    record = features.get(filename, {"filename": filename})
    record.update({
        "network_logo": data.network_logo,
        "logo_position": data.logo_position,
        "scoreboard_position": data.scoreboard_position,
    })
    features[filename] = record
    save_features(features)
    return {"saved": filename}


@router.get("/review")
def review(request: Request):
    save_dir = app_config.save_dir
    labels = load_labels()
    features = load_features()
    images = sorted(
        p.name
        for p in [*save_dir.glob("*.png"), *save_dir.glob("*.jpg")]
        if not p.name.startswith("compressed_")
    )
    image_data = [{"filename": f, "label": labels.get(f), "features": features.get(f, {})} for f in images]
    return templates.TemplateResponse(request, "review.html", {"image_data": image_data})
