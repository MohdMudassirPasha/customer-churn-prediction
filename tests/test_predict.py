from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.predict import clear_model_cache, load_model_artifact, predict_churn

VALID_CUSTOMER = {
    "customerID": "TEST-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 84.6,
    "TotalCharges": 423.0,
}


def _assert_valid_result(result: dict) -> None:
    assert result["prediction"] in (0, 1)
    assert result["prediction_label"] in ("Yes", "No")
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert 0.0 <= result["threshold"] <= 1.0
    assert isinstance(result["model_name"], str)


def test_load_model_artifact_returns_pipeline(fitted_model_artifact: Path):
    clear_model_cache()
    artifact = load_model_artifact(fitted_model_artifact)
    assert "pipeline" in artifact
    assert artifact["model_name"] == "Logistic Regression"


def test_predict_single_record(fitted_model_artifact: Path):
    clear_model_cache()
    results = predict_churn(VALID_CUSTOMER, model_path=fitted_model_artifact)
    assert len(results) == 1
    _assert_valid_result(results[0])


def test_predict_batch_dataframe(fitted_model_artifact: Path, sample_clean_data: pd.DataFrame):
    clear_model_cache()
    batch = sample_clean_data.drop(columns=["Churn"]).head(5)
    results = predict_churn(batch, model_path=fitted_model_artifact)
    assert len(results) == 5
    for result in results:
        _assert_valid_result(result)


def test_predict_list_of_records(fitted_model_artifact: Path):
    clear_model_cache()
    results = predict_churn([VALID_CUSTOMER, VALID_CUSTOMER], model_path=fitted_model_artifact)
    assert len(results) == 2
