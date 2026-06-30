"""Optuna-based hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import settings
from src.logger import get_logger
from src.pipeline import build_pipeline

logger = get_logger(__name__)


def _suggest_estimator(trial: Any, model_name: str) -> Any:
    """Create a trial estimator for the selected model family."""
    if model_name == "Logistic Regression":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 20.0, log=True),
            penalty="l2",
            solver="lbfgs",
            max_iter=1500,
            class_weight="balanced",
            random_state=settings.random_state,
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 120, 500),
            max_depth=trial.suggest_int("max_depth", 3, 18),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight="balanced",
            n_jobs=-1,
            random_state=settings.random_state,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 80, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            random_state=settings.random_state,
        )
    if model_name == "Support Vector Machine":
        return SVC(
            C=trial.suggest_float("C", 1e-2, 50.0, log=True),
            gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
            kernel=trial.suggest_categorical("kernel", ["rbf", "linear"]),
            probability=True,
            class_weight="balanced",
            random_state=settings.random_state,
        )
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 25),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
            class_weight="balanced",
            random_state=settings.random_state,
        )
    if model_name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=trial.suggest_int("n_neighbors", 3, 35),
            weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
            metric=trial.suggest_categorical("metric", ["minkowski", "manhattan"]),
        )
    if model_name == "XGBoost":
        xgboost = __import__("xgboost")
        return xgboost.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            eval_metric="logloss",
            n_jobs=-1,
            random_state=settings.random_state,
        )
    if model_name == "LightGBM":
        lightgbm = __import__("lightgbm")
        return lightgbm.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 80),
            max_depth=trial.suggest_int("max_depth", 3, 14),
            min_child_samples=trial.suggest_int("min_child_samples", 10, 80),
            class_weight="balanced",
            n_jobs=-1,
            random_state=settings.random_state,
            verbosity=-1,
        )
    if model_name == "CatBoost":
        catboost = __import__("catboost")
        return catboost.CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 100, 500),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=settings.random_state,
            verbose=False,
        )
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def optimize_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = settings.optuna_trials,
    cv_folds: int = settings.cv_folds,
    study_path: str | Path = settings.optuna_study_path,
) -> tuple[Any, Any]:
    """Optimize a model family with Optuna and return a fitted pipeline."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for hyperparameter tuning.") from exc

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=settings.random_state)

    def objective(trial: Any) -> float:
        estimator = _suggest_estimator(trial, model_name)
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="roc_auc",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        return float(scores.mean())

    sampler = optuna.samplers.TPESampler(seed=settings.random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_estimator = _suggest_estimator(study.best_trial, model_name)
    best_pipeline = build_pipeline(best_estimator)
    best_pipeline.fit(X_train, y_train)

    output_path = Path(study_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, output_path)
    logger.info("Saved Optuna study to %s", output_path)
    return best_pipeline, study
