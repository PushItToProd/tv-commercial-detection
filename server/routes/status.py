import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from pydantic import BaseModel

from config import app_config
from matrix import apply_matrix_settings
from state import last_image_path, sse_clients, state

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def _get_status_data() -> dict:
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
                except asyncio.TimeoutError:
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


class TriggerMatrixRequest(BaseModel):
    classification: str


@router.post("/trigger_matrix")
async def trigger_matrix(data: TriggerMatrixRequest):
    classification = data.classification
    if classification not in ("ad", "content"):
        raise HTTPException(status_code=400, detail="classification must be 'ad' or 'content'")
    await apply_matrix_settings(classification)
    await broadcast_status()
    return {"triggered": classification}
