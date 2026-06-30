"""Pipeline construction helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import settings
from src.feature_engineering import CustomerChurnFeatureEngineer
from src.preprocessing import DataFramePreprocessor


def build_pipeline(model: BaseEstimator | None = None) -> Pipeline:
    """Build a full training and inference pipeline."""
    # Use an explicit ``is None`` check: many sklearn estimators (e.g.
    # RandomForestClassifier) define ``__len__``/``__bool__``, so ``model or ...``
    # would evaluate their truthiness and raise before fitting.
    classifier = (
        model
        if model is not None
        else LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=settings.random_state,
        )
    )
    return Pipeline(
        steps=[
            ("feature_engineering", CustomerChurnFeatureEngineer()),
            ("preprocessor", DataFramePreprocessor()),
            ("model", classifier),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Return transformed feature names from a fitted pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    if preprocessor is None or not hasattr(preprocessor, "get_feature_names_out"):
        return []
    return [str(name) for name in preprocessor.get_feature_names_out()]


def get_model_step(pipeline: Pipeline) -> Any:
    """Return the estimator from a fitted pipeline."""
    if "model" not in pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")
    return pipeline.named_steps["model"]


def prepare_single_record(record: dict[str, Any]) -> pd.DataFrame:
    """Convert one JSON-like record to a dataframe suitable for inference."""
    return pd.DataFrame([record])
