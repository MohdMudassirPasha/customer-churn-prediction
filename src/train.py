"""Training entry point for the customer churn model."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from src.config import Settings, settings
from src.evaluate import generate_eda_figures, generate_shap_artifacts, save_evaluation_artifacts
from src.hyperparameter_tuning import optimize_model
from src.logger import get_logger, setup_logging
from src.model_selection import select_best_model, train_candidate_models
from src.pipeline import get_feature_names
from src.preprocessing import clean_telco_data, split_features_target
from src.utils import (
    build_model_card,
    ensure_project_directories,
    load_dataset,
    save_dataframe,
    save_joblib,
    save_json,
    set_global_seed,
)

logger = get_logger(__name__)


def _log_mlflow_run(
    model_name: str,
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    artifact_paths: list[Path],
    config: Settings = settings,
) -> None:
    """Track model training with MLflow."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("MLflow is not installed; skipping MLflow tracking.")
        return

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                if artifact_path.is_dir():
                    mlflow.log_artifacts(str(artifact_path))
                else:
                    mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(model, artifact_path="model")


def train_project(
    data_path: str | Path | None = None,
    allow_download: bool = False,
    tune: bool = True,
    generate_shap: bool = True,
    config: Settings = settings,
) -> dict[str, Any]:
    """Train, tune, evaluate, and persist the churn prediction model."""
    setup_logging(level=config.log_level, log_file=config.log_path)
    ensure_project_directories(config)
    set_global_seed(config.random_state)

    raw_data = load_dataset(data_path, allow_download=allow_download, config=config)
    cleaned = clean_telco_data(raw_data, target_column=config.target_column)
    save_dataframe(cleaned, config.processed_data_path)
    generate_eda_figures(cleaned, config.figures_dir, target_column=config.target_column)

    X, y = split_features_target(cleaned, config.target_column, config.id_column)
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.random_state,
    )
    valid_fraction = config.validation_size / (1.0 - config.test_size)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid,
        y_train_valid,
        test_size=valid_fraction,
        stratify=y_train_valid,
        random_state=config.random_state,
    )

    comparison, fitted_models = train_candidate_models(
        X_train,
        y_train,
        X_valid,
        y_valid,
        metric=config.model_selection_metric,
    )
    comparison.to_csv(config.comparison_path, index=False)
    best_name, best_pipeline = select_best_model(
        comparison,
        fitted_models,
        metric=config.model_selection_metric,
    )
    logger.info("Selected best model: %s", best_name)

    study = None
    if tune:
        try:
            best_pipeline, study = optimize_model(
                best_name,
                X_train_valid,
                y_train_valid,
                n_trials=config.optuna_trials,
                cv_folds=config.cv_folds,
                study_path=config.optuna_study_path,
            )
        except Exception as exc:
            logger.warning(
                "Hyperparameter tuning skipped; fitting selected model directly: %s", exc
            )
            best_pipeline.fit(X_train_valid, y_train_valid)
    else:
        best_pipeline.fit(X_train_valid, y_train_valid)

    evaluation = save_evaluation_artifacts(
        best_pipeline,
        X_test,
        y_test,
        output_dir=config.figures_dir,
        threshold=config.decision_threshold,
        metrics_path=config.metrics_path,
        classification_report_path=config.classification_report_path,
    )

    shap_artifacts: dict[str, str] = {}
    if generate_shap:
        try:
            shap_artifacts = generate_shap_artifacts(
                best_pipeline,
                X_train_valid,
                output_dir=config.shap_dir,
                sample_size=config.shap_sample_size,
            )
        except Exception as exc:
            logger.warning("SHAP artifacts skipped: %s", exc)

    feature_names = get_feature_names(best_pipeline)
    artifact = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "trained_at": datetime.now(UTC).isoformat(),
        "training_data_path": str(data_path or config.raw_data_path),
        "metrics": evaluation["metrics"],
        "classification_report": evaluation["classification_report"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "threshold": config.decision_threshold,
        "feature_names": feature_names,
        "raw_feature_names": X.columns.tolist(),
        "model_comparison": comparison.to_dict(orient="records"),
        "optuna_best_params": getattr(study, "best_params", None),
        "shap_artifacts": shap_artifacts,
    }
    save_joblib(artifact, config.model_path)

    model_card = build_model_card(
        best_name,
        evaluation["metrics"],
        feature_names,
        config.model_path,
        config.decision_threshold,
    )
    model_card["trained_at"] = artifact["trained_at"]
    model_card["optuna_best_params"] = artifact["optuna_best_params"]
    save_json(model_card, config.model_info_path)

    _log_mlflow_run(
        model_name=best_name,
        model=best_pipeline,
        metrics=evaluation["metrics"],
        params={
            "model_name": best_name,
            "tuned": tune,
            "threshold": config.decision_threshold,
            "test_size": config.test_size,
            "validation_size": config.validation_size,
        },
        artifact_paths=[
            config.model_path,
            config.model_info_path,
            config.metrics_path,
            config.comparison_path,
            config.figures_dir,
            config.shap_dir,
        ],
        config=config,
    )

    logger.info("Training complete. Model saved to %s", config.model_path)
    return artifact


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train customer churn prediction models.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw Telco churn CSV.")
    parser.add_argument("--download", action="store_true", help="Download the dataset if missing.")
    parser.add_argument("--no-tune", action="store_true", help="Skip Optuna tuning.")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP artifact generation.")
    return parser.parse_args()


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    train_project(
        data_path=args.data_path,
        allow_download=args.download,
        tune=not args.no_tune,
        generate_shap=not args.skip_shap,
    )


if __name__ == "__main__":
    main()
