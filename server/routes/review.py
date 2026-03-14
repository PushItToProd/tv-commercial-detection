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


@router.post("/save")
async def save(
    image: UploadFile = File(...),
    timestamp: str = Form(default=""),
    page_title: str = Form(default="?"),
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


class ClassifyRequest(BaseModel):
    filename: str
    label: str


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


@router.get("/review")
def review(request: Request):
    save_dir = app_config.save_dir
    labels = load_labels()
    images = sorted(
        p.name
        for p in [*save_dir.glob("*.png"), *save_dir.glob("*.jpg")]
        if not p.name.startswith("compressed_")
    )
    image_data = [{"filename": f, "label": labels.get(f)} for f in images]
    return templates.TemplateResponse(request, "review.html", {"image_data": image_data})
