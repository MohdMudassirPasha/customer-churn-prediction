from __future__ import annotations

from pathlib import Path

import pandas as pd

from dashboard import service
from src.predict import clear_model_cache


def test_build_customer_payload_derives_total_charges():
    payload = service.build_customer_payload(
        {"MonthlyCharges": 50.0, "tenure": 10, "TotalCharges": ""}
    )
    assert payload["TotalCharges"] == 500.0


def test_build_customer_payload_keeps_total_charges():
    payload = service.build_customer_payload(
        {"MonthlyCharges": 50.0, "tenure": 10, "TotalCharges": 123.0}
    )
    assert payload["TotalCharges"] == 123.0


def test_churn_rate_by_returns_rates(sample_clean_data: pd.DataFrame):
    result = service.churn_rate_by(sample_clean_data, "Contract")
    assert "churn_rate" in result.columns
    assert not result.empty
    assert (result["churn_rate"].between(0, 1)).all()


def test_churn_rate_by_missing_column(sample_clean_data: pd.DataFrame):
    result = service.churn_rate_by(sample_clean_data, "DoesNotExist")
    assert result.empty


def test_get_health_offline_returns_offline_status():
    health = service.get_health("http://127.0.0.1:59999")
    assert health["status"] == "offline"
    assert health["model_loaded"] is False


def test_local_prediction(fitted_model_artifact: Path):
    clear_model_cache()
    result = service.local_prediction(service.DEFAULT_CUSTOMER, model_path=fitted_model_artifact)
    assert result["prediction_label"] in ("Yes", "No")


def test_load_local_model_info(model_info_file: Path):
    info = service.load_local_model_info(model_info_file)
    assert info["model_name"] == "Logistic Regression"


def test_load_local_model_info_missing(tmp_path: Path):
    assert service.load_local_model_info(tmp_path / "missing.json") == {}
