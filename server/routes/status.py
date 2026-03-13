from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel

from config import app_config
from matrix import apply_matrix_settings
from state import last_image_path, state

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


@router.get("/is_ad/status")
def is_ad_status():
    output_settings = app_config.output_settings
    ad_view_label = output_settings.get("ad", {}).get("label", "Ad view")
    race_view_label = output_settings.get("content", {}).get("label", "Race view")

    return {
        "classification": state.classification,
        "paused": state.paused,
        "seeking": state.seeking,
        "pending": state.is_pending_change(),
        "auto_switch": state.auto_switch,
        "enable_debounce": state.enable_debounce,
        "ad_view_label": ad_view_label,
        "race_view_label": race_view_label,
    }


@router.get("/is_ad")
def is_ad(request: Request):
    return templates.TemplateResponse(request, "is_ad.html")


@router.get("/is_ad/last_frame")
def last_frame():
    if not last_image_path.exists():
        return Response(status_code=204)
    return FileResponse(last_image_path, media_type="image/jpeg")


class EnabledRequest(BaseModel):
    enabled: bool


@router.post("/auto_switch")
def set_auto_switch(data: EnabledRequest):
    state.auto_switch = data.enabled
    return {"auto_switch": state.auto_switch}


@router.post("/enable_debounce")
def set_enable_debounce(data: EnabledRequest):
    state.enable_debounce = data.enabled
    return {"enable_debounce": state.enable_debounce}


class TriggerMatrixRequest(BaseModel):
    classification: str


@router.post("/trigger_matrix")
def trigger_matrix(data: TriggerMatrixRequest):
    classification = data.classification
    if classification not in ("ad", "content"):
        raise HTTPException(status_code=400, detail="classification must be 'ad' or 'content'")
    apply_matrix_settings(classification)
    return {"triggered": classification}

