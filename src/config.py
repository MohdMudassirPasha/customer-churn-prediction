"""Application configuration for the customer churn prediction project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: Path) -> Path:
    """Return an absolute path from an environment variable or a default path."""
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()
    return default.resolve()


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings."""

    project_name: str = "Customer Churn Prediction"
    environment: str = os.getenv("APP_ENV", "development")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    validation_size: float = float(os.getenv("VALIDATION_SIZE", "0.2"))
    target_column: str = "Churn"
    id_column: str = "customerID"
    positive_label: str = "Yes"
    negative_label: str = "No"
    decision_threshold: float = float(os.getenv("CHURN_DECISION_THRESHOLD", "0.5"))
    model_selection_metric: str = os.getenv("MODEL_SELECTION_METRIC", "roc_auc")

    data_url: str = os.getenv(
        "TELCO_CHURN_DATA_URL",
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    )
    raw_data_path: Path = field(
        default_factory=lambda: _path_from_env(
            "CHURN_RAW_DATA",
            PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        )
    )
    sample_data_path: Path = field(
        default_factory=lambda: _path_from_env(
            "CHURN_SAMPLE_DATA",
            PROJECT_ROOT / "data" / "raw" / "sample_telco_churn.csv",
        )
    )
    processed_data_path: Path = field(
        default_factory=lambda: _path_from_env(
            "CHURN_PROCESSED_DATA",
            PROJECT_ROOT / "data" / "processed" / "telco_churn_processed.csv",
        )
    )

    model_dir: Path = field(
        default_factory=lambda: _path_from_env("MODEL_DIR", PROJECT_ROOT / "models")
    )
    model_path: Path = field(
        default_factory=lambda: _path_from_env(
            "MODEL_PATH",
            PROJECT_ROOT / "models" / "churn_model.joblib",
        )
    )
    model_info_path: Path = field(
        default_factory=lambda: _path_from_env(
            "MODEL_INFO_PATH",
            PROJECT_ROOT / "models" / "model_info.json",
        )
    )
    optuna_study_path: Path = field(
        default_factory=lambda: _path_from_env(
            "OPTUNA_STUDY_PATH",
            PROJECT_ROOT / "models" / "optuna_study.joblib",
        )
    )

    reports_dir: Path = field(
        default_factory=lambda: _path_from_env("REPORTS_DIR", PROJECT_ROOT / "reports")
    )
    figures_dir: Path = field(
        default_factory=lambda: _path_from_env(
            "FIGURES_DIR",
            PROJECT_ROOT / "reports" / "figures",
        )
    )
    shap_dir: Path = field(
        default_factory=lambda: _path_from_env("SHAP_DIR", PROJECT_ROOT / "reports" / "shap")
    )
    metrics_path: Path = field(
        default_factory=lambda: _path_from_env(
            "METRICS_PATH",
            PROJECT_ROOT / "reports" / "metrics.json",
        )
    )
    comparison_path: Path = field(
        default_factory=lambda: _path_from_env(
            "MODEL_COMPARISON_PATH",
            PROJECT_ROOT / "reports" / "model_comparison.csv",
        )
    )
    classification_report_path: Path = field(
        default_factory=lambda: _path_from_env(
            "CLASSIFICATION_REPORT_PATH",
            PROJECT_ROOT / "reports" / "classification_report.json",
        )
    )

    mlflow_enabled: bool = os.getenv("MLFLOW_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    mlflow_experiment_name: str = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "customer-churn-prediction",
    )
    mlflow_tracking_uri: str = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}",
    )

    optuna_trials: int = int(os.getenv("OPTUNA_TRIALS", "50"))
    cv_folds: int = int(os.getenv("CV_FOLDS", "3"))
    shap_sample_size: int = int(os.getenv("SHAP_SAMPLE_SIZE", "250"))

    api_title: str = "Customer Churn Prediction API"
    api_version: str = "1.0.0"
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))
    dashboard_api_url: str = os.getenv("API_URL", "http://localhost:8000")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_path: Path = field(
        default_factory=lambda: _path_from_env("LOG_FILE", PROJECT_ROOT / "logs" / "app.log")
    )

    def ensure_directories(self) -> None:
        """Create expected runtime directories."""
        for path in (
            self.raw_data_path.parent,
            self.processed_data_path.parent,
            self.model_dir,
            self.reports_dir,
            self.figures_dir,
            self.shap_dir,
            self.log_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
