"""Model registry and automatic model selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import settings
from src.evaluate import evaluate_classifier
from src.logger import get_logger
from src.pipeline import build_pipeline

logger = get_logger(__name__)
ModelFactory = Callable[[], Any]


def _optional_model(name: str, factory: Callable[[], Any]) -> tuple[str, ModelFactory] | None:
    """Return an optional model factory only when its dependency imports cleanly."""
    try:
        model = factory()
    except ImportError:
        logger.warning("%s is not installed and will be skipped.", name)
        return None
    return name, lambda: model


def get_model_registry(random_state: int = settings.random_state) -> dict[str, ModelFactory]:
    """Return all supported model factories."""
    registry: dict[str, ModelFactory] = {
        "Logistic Regression": lambda: LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=random_state),
        "Support Vector Machine": lambda: SVC(
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state,
        ),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=15),
    }

    optional_factories = [
        _optional_model(
            "XGBoost",
            lambda: __import__("xgboost").XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        _optional_model(
            "LightGBM",
            lambda: __import__("lightgbm").LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        _optional_model(
            "CatBoost",
            lambda: __import__("catboost").CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=random_state,
                verbose=False,
            ),
        ),
    ]
    for item in optional_factories:
        if item is not None:
            name, factory = item
            registry[name] = factory
    return registry


def train_candidate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric: str = settings.model_selection_metric,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train all candidate models and return a ranked comparison table."""
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}

    for name, factory in get_model_registry().items():
        logger.info("Training candidate model: %s", name)
        pipeline = build_pipeline(factory())
        try:
            pipeline.fit(X_train, y_train)
            evaluation = evaluate_classifier(pipeline, X_valid, y_valid)
            row = {"model_name": name, **evaluation["metrics"]}
            rows.append(row)
            fitted_models[name] = pipeline
        except Exception as exc:
            logger.exception("Candidate model %s failed: %s", name, exc)
            rows.append({"model_name": name, "error": str(exc)})

    comparison = pd.DataFrame(rows)
    if comparison.empty:
        raise RuntimeError("No models were trained successfully.")
    if metric not in comparison.columns:
        raise ValueError(f"Metric '{metric}' is not available in model comparison.")

    comparison = comparison.sort_values(metric, ascending=False, na_position="last").reset_index(
        drop=True
    )
    return comparison, fitted_models


def select_best_model(
    comparison: pd.DataFrame,
    fitted_models: dict[str, Any],
    metric: str = settings.model_selection_metric,
) -> tuple[str, Any]:
    """Select the top model from a comparison table."""
    if comparison.empty:
        raise ValueError("Model comparison is empty.")
    successful = comparison.dropna(subset=[metric])
    if successful.empty:
        raise RuntimeError("No successful model has the requested selection metric.")
    best_name = str(successful.iloc[0]["model_name"])
    if best_name not in fitted_models:
        raise RuntimeError(f"Best model '{best_name}' was not fitted.")
    return best_name, fitted_models[best_name]
