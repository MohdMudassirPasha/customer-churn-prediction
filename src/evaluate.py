"""Evaluation, EDA, and explainability utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.config import settings
from src.logger import get_logger
from src.pipeline import get_feature_names, get_model_step
from src.utils import save_json

logger = get_logger(__name__)


def prediction_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class prediction scores for a classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(X)
        minimum = np.min(raw_scores)
        maximum = np.max(raw_scores)
        if maximum == minimum:
            return np.zeros_like(raw_scores, dtype=float)
        return (raw_scores - minimum) / (maximum - minimum)
    return model.predict(X).astype(float)


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["roc_auc"] = 0.0
    return {key: float(value) for key, value in metrics.items()}


def evaluate_classifier(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: float = settings.decision_threshold,
) -> dict[str, Any]:
    """Evaluate a fitted classifier and return metrics plus report data."""
    scores = prediction_scores(model, X)
    predictions = (scores >= threshold).astype(int)
    metrics = compute_metrics(y, predictions, scores)
    report = classification_report(y, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y, predictions).tolist()
    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "threshold": threshold,
    }


def save_evaluation_artifacts(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    output_dir: str | Path = settings.figures_dir,
    threshold: float = settings.decision_threshold,
    metrics_path: str | Path = settings.metrics_path,
    classification_report_path: str | Path = settings.classification_report_path,
) -> dict[str, Any]:
    """Generate and save model evaluation plots and JSON artifacts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = evaluate_classifier(model, X, y, threshold=threshold)
    scores = prediction_scores(model, X)
    predictions = (scores >= threshold).astype(int)

    save_json(result["metrics"], metrics_path)
    save_json(result["classification_report"], classification_report_path)

    ConfusionMatrixDisplay.from_predictions(y, predictions, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y, scores)
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path / "roc_curve.png", dpi=160)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y, scores)
    plt.title("Precision Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path / "precision_recall_curve.png", dpi=160)
    plt.close()

    save_feature_importance_plot(model, output_path / "feature_importance.png")
    return result


def _target_values(data: pd.DataFrame, target_column: str) -> pd.Series:
    """Normalize target values for plots."""
    target = data[target_column]
    if target.dtype.kind in {"i", "u", "f", "b"}:
        return target.astype(int)
    return target.astype(str).str.strip().map({"No": 0, "Yes": 1}).fillna(0).astype(int)


def _save_bar(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    path: Path,
    color: str = "#2f80ed",
) -> None:
    """Save a compact bar chart."""
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=color)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_eda_figures(
    data: pd.DataFrame,
    output_dir: str | Path = settings.figures_dir,
    target_column: str = settings.target_column,
) -> dict[str, str]:
    """Generate professional EDA figures for the Telco churn dataset."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is required for EDA.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}
    target = _target_values(data, target_column)

    counts = target.value_counts().reindex([0, 1], fill_value=0)
    class_path = output_path / "class_imbalance.png"
    _save_bar(["No Churn", "Churn"], counts.tolist(), "Class Imbalance", "Customers", class_path)
    plot_paths["class_imbalance"] = str(class_path)

    numeric_columns = [
        column for column in ["MonthlyCharges", "TotalCharges", "tenure"] if column in data.columns
    ]
    for column in numeric_columns:
        numeric = pd.to_numeric(data[column], errors="coerce").dropna()
        plt.figure(figsize=(8, 5))
        plt.hist(numeric, bins=30, color="#27ae60", edgecolor="white", alpha=0.9)
        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Customers")
        plt.tight_layout()
        figure_path = output_path / f"{column.lower()}_distribution.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        plot_paths[f"{column.lower()}_distribution"] = str(figure_path)

    for column in ["Contract", "InternetService", "PaymentMethod"]:
        if column not in data.columns:
            continue
        churn_rate = (
            data.assign(_target=target)
            .groupby(column, dropna=False)["_target"]
            .mean()
            .sort_values(ascending=False)
        )
        figure_path = output_path / f"{column.lower()}_churn_rate.png"
        _save_bar(
            [str(index) for index in churn_rate.index],
            churn_rate.round(4).tolist(),
            f"Churn Rate by {column}",
            "Churn Rate",
            figure_path,
            color="#eb5757",
        )
        plot_paths[f"{column.lower()}_churn_rate"] = str(figure_path)

    numeric_frame = data.copy()
    for column in numeric_frame.columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    numeric_frame[target_column] = target
    numeric_frame = numeric_frame.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    if numeric_frame.shape[1] >= 2:
        corr = numeric_frame.corr(numeric_only=True)
        plt.figure(figsize=(9, 7))
        image = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(image, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        figure_path = output_path / "correlation_heatmap.png"
        plt.savefig(figure_path, dpi=160)
        plt.close()
        plot_paths["correlation_heatmap"] = str(figure_path)

    return plot_paths


def extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    """Extract feature importance or coefficient magnitude from a fitted model."""
    estimator = get_model_step(model)
    feature_names = get_feature_names(model)

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if not feature_names or len(feature_names) != len(importance):
        feature_names = [f"feature_{index}" for index in range(len(importance))]

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_plot(model: Pipeline, path: str | Path, top_n: int = 20) -> Path | None:
    """Save a feature importance chart when the estimator supports it."""
    importance = extract_feature_importance(model)
    if importance.empty:
        return None

    top_features = importance.head(top_n).iloc[::-1]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="#9b51e0")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def generate_shap_artifacts(
    model: Pipeline,
    X: pd.DataFrame,
    output_dir: str | Path = settings.shap_dir,
    sample_size: int = settings.shap_sample_size,
) -> dict[str, str]:
    """Generate SHAP explainability artifacts for a fitted pipeline."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required to generate explainability artifacts.") from exc

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sample = X.sample(min(sample_size, len(X)), random_state=settings.random_state)
    engineered = model.named_steps["feature_engineering"].transform(sample)
    transformed = model.named_steps["preprocessor"].transform(engineered)
    feature_names = get_feature_names(model)
    estimator = get_model_step(model)

    if hasattr(estimator, "predict_proba"):

        def predictor(matrix: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(matrix)[:, 1]

    else:
        predictor = estimator.predict

    explainer = shap.Explainer(predictor, transformed, feature_names=feature_names)
    shap_values = explainer(transformed)
    shap_array = np.asarray(shap_values.values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]

    artifacts: dict[str, str] = {}
    plt.figure()
    shap.summary_plot(shap_array, transformed, feature_names=feature_names, show=False)
    summary_path = output_path / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, dpi=160, bbox_inches="tight")
    plt.close()
    artifacts["summary_plot"] = str(summary_path)

    mean_importance = np.abs(shap_array).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": mean_importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_path = output_path / "shap_feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    artifacts["feature_importance"] = str(importance_path)

    top_features = importance.head(3)["feature"].tolist()
    transformed_frame = pd.DataFrame(transformed, columns=feature_names)
    for feature in top_features:
        figure_path = output_path / f"shap_dependence_{feature[:40].replace(' ', '_')}.png"
        plt.figure()
        shap.dependence_plot(
            feature,
            shap_array,
            transformed_frame,
            feature_names=feature_names,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(figure_path, dpi=160, bbox_inches="tight")
        plt.close()
        artifacts[f"dependence_{feature}"] = str(figure_path)

    first = shap.Explanation(
        values=shap_array[0],
        base_values=np.asarray(shap_values.base_values).reshape(-1)[0],
        data=transformed[0],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(first, max_display=15, show=False)
    waterfall_path = output_path / "shap_waterfall.png"
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=160, bbox_inches="tight")
    plt.close()
    artifacts["waterfall_plot"] = str(waterfall_path)

    metadata_path = output_path / "shap_artifacts.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(artifacts, file, indent=2)
    artifacts["metadata"] = str(metadata_path)
    return artifacts
