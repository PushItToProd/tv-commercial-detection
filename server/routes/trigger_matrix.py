from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from routes.status import broadcast_status
import matrix
from state import state


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
        raise HTTPException(status_code=400, detail="classification must be 'ad' or 'content'")
    await apply_matrix_settings(classification)
    return {"triggered": classification}
