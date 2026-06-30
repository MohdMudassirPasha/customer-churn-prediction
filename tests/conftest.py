from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import Settings
from src.pipeline import build_pipeline
from src.preprocessing import clean_telco_data, split_features_target
from src.utils import save_joblib, save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sample_telco_churn.csv"


@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA_PATH)


@pytest.fixture
def sample_clean_data(sample_raw_data: pd.DataFrame) -> pd.DataFrame:
    return clean_telco_data(sample_raw_data)


@pytest.fixture
def fitted_model_artifact(sample_clean_data: pd.DataFrame, tmp_path: Path) -> Path:
    X, y = split_features_target(sample_clean_data)
    pipeline = build_pipeline(
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    )
    pipeline.fit(X, y)
    artifact = {
        "pipeline": pipeline,
        "model_name": "Logistic Regression",
        "threshold": 0.5,
        "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
        "feature_names": [],
    }
    model_path = tmp_path / "models" / "churn_model.joblib"
    save_joblib(artifact, model_path)
    return model_path


@pytest.fixture
def temp_settings(tmp_path: Path) -> Settings:
    return replace(
        Settings(),
        raw_data_path=SAMPLE_DATA_PATH,
        sample_data_path=SAMPLE_DATA_PATH,
        processed_data_path=tmp_path / "data" / "processed" / "telco_churn_processed.csv",
        model_dir=tmp_path / "models",
        model_path=tmp_path / "models" / "churn_model.joblib",
        model_info_path=tmp_path / "models" / "model_info.json",
        optuna_study_path=tmp_path / "models" / "optuna_study.joblib",
        reports_dir=tmp_path / "reports",
        figures_dir=tmp_path / "reports" / "figures",
        shap_dir=tmp_path / "reports" / "shap",
        metrics_path=tmp_path / "reports" / "metrics.json",
        comparison_path=tmp_path / "reports" / "model_comparison.csv",
        classification_report_path=tmp_path / "reports" / "classification_report.json",
        log_path=tmp_path / "logs" / "app.log",
        optuna_trials=2,
        shap_sample_size=10,
    )


@pytest.fixture
def model_info_file(temp_settings: Settings, fitted_model_artifact: Path) -> Path:
    payload = {
        "model_name": "Logistic Regression",
        "artifact_path": str(fitted_model_artifact),
        "decision_threshold": 0.5,
        "metrics": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0},
        "feature_count": 0,
        "features": [],
        "trained_at": "2026-06-30T00:00:00+00:00",
        "optuna_best_params": None,
    }
    save_json(payload, temp_settings.model_info_path)
    return temp_settings.model_info_path
