import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from ..classify import list_profiles
from ..config import app_config
from ..state import last_image_path, sse_clients, state

router = APIRouter()

templates = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)


def _get_status_data() -> dict:
    output_settings = app_config.output_settings
    ad_view_label = output_settings.get("ad", {}).get("label", "Ad view")
    race_view_label = output_settings.get("content", {}).get("label", "Race view")
    return {
        "classification": state.classification,
        "classification_reason": state.classification_reason,
        "paused": state.paused,
        "seeking": state.seeking,
        "pending": state.is_pending_change(),
        "auto_switch": state.auto_switch,
        "enable_debounce": state.enable_debounce,
        "matrix_switching": state.matrix_switching,
        "ad_view_label": ad_view_label,
        "race_view_label": race_view_label,
        "auto_switch_paused_until": state.auto_switch_paused_until
        if state.is_auto_switch_paused()
        else None,
    }


async def broadcast_status() -> None:
    if not sse_clients:
        return
    data = _get_status_data()
    for q in list(sse_clients):
        await q.put(data)


@router.get("/is_ad/status")
def is_ad_status():
    return _get_status_data()


@router.get("/is_ad/stream")
async def is_ad_stream():
    queue: asyncio.Queue = asyncio.Queue()
    sse_clients.add(queue)

    async def event_generator():
        try:
            yield f"data: {json.dumps(_get_status_data())}\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {json.dumps(data)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            sse_clients.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@router.post("/settings/auto_switch")
async def set_auto_switch(data: EnabledRequest):
    state.auto_switch = data.enabled
    await broadcast_status()
    return {"auto_switch": state.auto_switch}


@router.post("/settings/enable_debounce")
async def set_enable_debounce(data: EnabledRequest):
    state.enable_debounce = data.enabled
    await broadcast_status()
    return {"enable_debounce": state.enable_debounce}


@router.get("/settings/classifier_profile")
def get_classifier_profile():
    return {"current": app_config.classifier_profile, "available": list_profiles()}


class ClassifierProfileRequest(BaseModel):
    profile: str


@router.post("/settings/classifier_profile")
async def set_classifier_profile(data: ClassifierProfileRequest):

    # if not re.fullmatch(r"[a-z][a-z0-9_]*", data.profile):
    #     from fastapi import HTTPException
    #     raise HTTPException(status_code=422, detail="Invalid profile name")
    if data.profile not in list_profiles():
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail=f"Profile '{data.profile}' not found")
    app_config.classifier_profile = data.profile
    await broadcast_status()
    return {"classifier_profile": app_config.classifier_profile}
