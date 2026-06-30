"""Prediction utilities for trained churn models."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import settings
from src.evaluate import prediction_scores
from src.utils import load_joblib


def _as_dataframe(records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Convert supported prediction inputs to a dataframe."""
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame([records])
    if isinstance(records, list):
        if not all(isinstance(record, dict) for record in records):
            raise TypeError("Prediction list inputs must contain dictionaries.")
        return pd.DataFrame(records)
    raise TypeError("Prediction input must be a dictionary, list of dictionaries, or dataframe.")


@lru_cache(maxsize=2)
def load_model_artifact(model_path: str | Path = settings.model_path) -> dict[str, Any]:
    """Load a persisted model artifact."""
    artifact = load_joblib(model_path)
    if isinstance(artifact, Pipeline):
        return {
            "pipeline": artifact,
            "model_name": artifact.named_steps["model"].__class__.__name__,
            "threshold": settings.decision_threshold,
        }
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError("Model artifact must be a pipeline or a dictionary containing 'pipeline'.")
    return artifact


def predict_churn(
    records: dict[str, Any] | list[dict[str, Any]] | pd.DataFrame,
    model_path: str | Path = settings.model_path,
) -> list[dict[str, Any]]:
    """Predict churn probabilities and labels for one or more records."""
    artifact = load_model_artifact(model_path)
    pipeline = artifact["pipeline"]
    threshold = float(artifact.get("threshold", settings.decision_threshold))
    data = _as_dataframe(records)
    scores = prediction_scores(pipeline, data)
    predictions = (scores >= threshold).astype(int)

    results = []
    for index, score in enumerate(np.asarray(scores, dtype=float)):
        prediction = int(predictions[index])
        results.append(
            {
                "prediction": prediction,
                "prediction_label": settings.positive_label
                if prediction == 1
                else settings.negative_label,
                "churn_probability": float(score),
                "threshold": threshold,
                "model_name": artifact.get("model_name", "unknown"),
            }
        )
    return results


def clear_model_cache() -> None:
    """Clear cached model artifacts."""
    load_model_artifact.cache_clear()
