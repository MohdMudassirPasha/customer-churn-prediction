"""API routes for the customer churn prediction service."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResponse, TelcoCustomer
from src.config import settings
from src.predict import load_model_artifact, predict_churn
from src.utils import load_json

router = APIRouter()


@router.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return basic service information."""
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Return service health and model availability."""
    try:
        load_model_artifact(settings.model_path)
        return HealthResponse(status="ok", model_loaded=True, model_path=str(settings.model_path))
    except Exception:
        return HealthResponse(
            status="degraded", model_loaded=False, model_path=str(settings.model_path)
        )


@router.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: TelcoCustomer) -> PredictionResponse:
    """Predict churn for a validated customer payload."""
    try:
        result = predict_churn(payload.to_model_record(), settings.model_path)[0]
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not available. Train the model before serving predictions.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc
    return PredictionResponse(**result)


@router.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Return metadata for the currently deployed model."""
    try:
        payload = load_json(settings.model_info_path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is not available. Train the model before requesting model info.",
        ) from exc
    return ModelInfoResponse(**payload)
