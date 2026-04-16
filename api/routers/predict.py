"""POST /api/predict — HARE model inference."""

from fastapi import APIRouter, Depends, File, UploadFile

from ..core.security import get_current_user
from ..domain.schemas import PredictionResponse
from ..services.prediction_service import prediction_service

router = APIRouter(prefix="/api", tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    _user: dict = Depends(get_current_user),
) -> PredictionResponse:
    """Runs HARE prediction on an uploaded dermoscopic image."""
    image_bytes = await image.read()
    return prediction_service.predict(image_bytes)
