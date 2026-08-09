import json
import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel

from ..config import app_config

logger = logging.getLogger(__name__)
router = APIRouter()

templates = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)


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
VALID_NETWORK_LOGOS = frozenset(
    {"Fox", "FS1", "FS2", "NBC", "CW", "USA", "Prime", "TNT", "other", "none"}
)
VALID_LOGO_POSITIONS = frozenset(
    {
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "center",
        "not_visible",
        "unknown",
    }
)
VALID_SCOREBOARD_POSITIONS = frozenset(
    {"top", "bottom", "left", "upper_left", "right", "none", "unknown"}
)

VALID_LABELS = frozenset({"ad", "content", "ignore"})

# Feature fields that /review can filter on, mapped to their allowed values.
FILTERABLE_FEATURES = {
    "network_logo": VALID_NETWORK_LOGOS,
    "logo_position": VALID_LOGO_POSITIONS,
    "scoreboard_position": VALID_SCOREBOARD_POSITIONS,
}

# Filter sentinel meaning "field has no value recorded". It can't collide with a
# real value the way "none" would (both network_logo and scoreboard_position
# accept a literal "none").
UNSET = "__unset__"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
DEFAULT_PER_PAGE = 100
MAX_PER_PAGE = 500


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
        dt = (
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if timestamp
            else datetime.now()
        )
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
    # Guard against path traversal: filename must be a plain basename
    # ending in .jpg or .png
    if Path(filename).name != filename or not filename.endswith(
        (".jpg", ".jpeg", ".png")
    ):
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
    if Path(filename).name != filename or not filename.endswith(
        (".jpg", ".jpeg", ".png")
    ):
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
        raise HTTPException(
            status_code=400, detail="label must be 'ad', 'content', or 'ignore'"
        )

    filename = data.filename
    # Guard against path traversal: filename must be a plain basename
    # ending in .png or .jpg
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
    if (
        data.logo_position is not None
        and data.logo_position not in VALID_LOGO_POSITIONS
    ):
        raise HTTPException(status_code=400, detail="Invalid logo_position value")
    if (
        data.scoreboard_position is not None
        and data.scoreboard_position not in VALID_SCOREBOARD_POSITIONS
    ):
        raise HTTPException(status_code=400, detail="Invalid scoreboard_position value")

    filename = data.filename
    if Path(filename).name != filename or not filename.endswith(
        (".png", ".jpg", ".jpeg")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    features = load_features()
    record = features.get(filename, {"filename": filename})
    record.update(
        {
            "network_logo": data.network_logo,
            "logo_position": data.logo_position,
            "scoreboard_position": data.scoreboard_position,
        }
    )
    features[filename] = record
    save_features(features)
    return {"saved": filename}


def list_frame_names() -> list[str]:
    """All reviewable frame filenames in the save dir, sorted by name.

    Uses os.scandir rather than Path.glob: the save dir holds tens of thousands
    of frames and scandir avoids building a Path object per entry.
    """
    try:
        entries = os.scandir(app_config.save_dir)
    except FileNotFoundError:
        return []
    with entries:
        return sorted(
            e.name
            for e in entries
            if e.name.endswith(IMAGE_SUFFIXES)
            and not e.name.startswith("compressed_")
            and e.is_file()
        )


def _validate_choice(name: str, value: str, allowed: frozenset[str]) -> None:
    if value and value != UNSET and value not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid {name} filter: {value!r}")


def _matches(value: str | None, wanted: str) -> bool:
    """Whether a stored field value satisfies a filter selection."""
    if not wanted:
        return True
    if wanted == UNSET:
        return not value
    return value == wanted


def _is_incomplete(label: str | None, feats: dict) -> bool:
    return not label or not all(feats.get(f) for f in FILTERABLE_FEATURES)


@router.get("/review")
def review(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(DEFAULT_PER_PAGE, ge=1, le=MAX_PER_PAGE),
    sort: str = Query("asc", pattern="^(asc|desc)$"),
    label: str = "",
    network_logo: str = "",
    logo_position: str = "",
    scoreboard_position: str = "",
    q: str = "",
    start: str = "",
    end: str = "",
    incomplete: bool = False,
):
    """Paginated, filterable frame review UI.

    Every filter and the page number live in query params so any view is
    linkable and bookmarkable.
    """
    feature_filters = {
        "network_logo": network_logo,
        "logo_position": logo_position,
        "scoreboard_position": scoreboard_position,
    }
    _validate_choice("label", label, VALID_LABELS)
    for field_name, value in feature_filters.items():
        _validate_choice(field_name, value, FILTERABLE_FEATURES[field_name])

    labels = load_labels()
    features = load_features()

    names = list_frame_names()
    if sort == "desc":
        names.reverse()

    # Filenames are timestamp-prefixed (YYYY-MM-DD...), so date bounds are a
    # plain lexicographic comparison on the leading 10 characters.
    q_lower = q.lower()
    matched = []
    for name in names:
        if q_lower and q_lower not in name.lower():
            continue
        if start and name[:10] < start:
            continue
        if end and name[:10] > end:
            continue
        item_label = labels.get(name)
        if not _matches(item_label, label):
            continue
        feats = features.get(name, {})
        if not all(
            _matches(feats.get(f), wanted) for f, wanted in feature_filters.items()
        ):
            continue
        if incomplete and not _is_incomplete(item_label, feats):
            continue
        matched.append((name, item_label, feats))

    total = len(matched)
    total_pages = max(1, -(-total // per_page))
    page = min(page, total_pages)
    offset = (page - 1) * per_page
    image_data = [
        {"filename": name, "label": item_label, "features": feats}
        for name, item_label, feats in matched[offset : offset + per_page]
    ]

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "image_data": image_data,
            "filters": {
                "label": label,
                **feature_filters,
                "q": q,
                "start": start,
                "end": end,
                "incomplete": incomplete,
                "sort": sort,
            },
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "offset": offset,
            },
            "unset_value": UNSET,
            "default_per_page": DEFAULT_PER_PAGE,
        },
    )
