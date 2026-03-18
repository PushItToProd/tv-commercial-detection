from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import matrix
from ..state import state
from .status import broadcast_status

router = APIRouter()


async def apply_matrix_settings(classification: str) -> None:
    state.matrix_switching = True
    await broadcast_status()

    try:
        await matrix.apply_matrix_settings(classification)
    finally:
        state.matrix_switching = False
        await broadcast_status()


class TriggerMatrixRequest(BaseModel):
    classification: str


@router.post("/trigger_matrix")
async def trigger_matrix(data: TriggerMatrixRequest):
    classification = data.classification
    if classification not in ("ad", "content"):
        raise HTTPException(
            status_code=400, detail="classification must be 'ad' or 'content'"
        )
    # Disable auto-switch since I switched manually
    state.auto_switch = False
    await apply_matrix_settings(classification)
    return {"triggered": classification}


@router.post("/settings/resume_auto_switch")
async def resume_auto_switch():
    """Clear the temporary auto-switch pause and immediately switch
    to the correct input."""
    state.auto_switch_paused_until = None
    state.auto_switch = True
    await broadcast_status()
    if state.classification in ("ad", "content"):
        await apply_matrix_settings(state.classification)
    return {"auto_switch": True, "auto_switch_paused_until": None}
