"""Shared dashboard helpers that are easy to test without launching Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import settings
from src.predict import predict_churn
from src.utils import load_dataset, load_json, load_sample_dataset

DEFAULT_CUSTOMER: dict[str, Any] = {
    "customerID": "SAMPLE-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.6,
    "TotalCharges": 1015.2,
}


def build_customer_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Return an API-ready Telco payload with derived total charges when needed."""
    payload = {**DEFAULT_CUSTOMER, **values}
    monthly = float(payload["MonthlyCharges"])
    tenure = int(payload["tenure"])
    total = payload.get("TotalCharges")
    if total in (None, ""):
        total = monthly * tenure
    payload["MonthlyCharges"] = monthly
    payload["tenure"] = tenure
    payload["TotalCharges"] = float(total)
    return payload


def request_prediction(
    payload: dict[str, Any], api_url: str = settings.dashboard_api_url
) -> dict[str, Any]:
    """Request a prediction from the FastAPI service."""
    response = requests.post(f"{api_url.rstrip('/')}/predict", json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def local_prediction(
    payload: dict[str, Any], model_path: str | Path = settings.model_path
) -> dict[str, Any]:
    """Run a prediction directly from the persisted local model artifact."""
    return predict_churn(payload, model_path=model_path)[0]


def get_health(api_url: str = settings.dashboard_api_url) -> dict[str, Any]:
    """Return API health information, or an offline status if the API is unreachable."""
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"status": "offline", "model_loaded": False, "error": str(exc)}


def get_model_info(api_url: str = settings.dashboard_api_url) -> dict[str, Any]:
    """Return model info from the API when available."""
    response = requests.get(f"{api_url.rstrip('/')}/model-info", timeout=5)
    response.raise_for_status()
    return response.json()


def load_local_model_info(path: str | Path = settings.model_info_path) -> dict[str, Any]:
    """Load model info from disk if it exists."""
    info_path = Path(path)
    if not info_path.exists():
        return {}
    return load_json(info_path)


def load_dashboard_dataset() -> pd.DataFrame:
    """Load processed data, raw data, or the bundled sample for dashboard charts."""
    if settings.processed_data_path.exists():
        return pd.read_csv(settings.processed_data_path)
    try:
        return load_dataset(allow_download=False)
    except FileNotFoundError:
        return load_sample_dataset()


def churn_rate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate churn rate by a categorical column."""
    if column not in data.columns or settings.target_column not in data.columns:
        return pd.DataFrame(columns=[column, "churn_rate"])
    target = data[settings.target_column]
    if target.dtype.kind not in {"i", "u", "f", "b"}:
        target = target.astype(str).str.strip().map({"No": 0, "Yes": 1}).fillna(0).astype(int)
    grouped = (
        data.assign(_target=target)
        .groupby(column, dropna=False)["_target"]
        .mean()
        .reset_index(name="churn_rate")
        .sort_values("churn_rate", ascending=False)
    )
    return grouped
