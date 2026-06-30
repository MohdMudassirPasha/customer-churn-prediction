from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.pipeline import build_pipeline, get_feature_names, prepare_single_record
from src.preprocessing import split_features_target


def test_build_pipeline_default_is_logistic_regression():
    pipeline = build_pipeline()
    assert pipeline.named_steps["model"].__class__.__name__ == "LogisticRegression"


def test_build_pipeline_accepts_estimator_defining_len():
    """Regression guard: estimators with ``__len__`` (e.g. RandomForest) must not
    trigger truthiness evaluation inside ``build_pipeline``."""
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    assert pipeline.named_steps["model"].__class__.__name__ == "RandomForestClassifier"


def test_get_feature_names_after_fit(sample_clean_data):
    X, y = split_features_target(sample_clean_data)
    pipeline = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipeline.fit(X, y)

    names = get_feature_names(pipeline)
    assert isinstance(names, list)
    assert len(names) > 0


def test_prepare_single_record_returns_single_row():
    frame = prepare_single_record({"a": 1, "b": "x"})
    assert frame.shape[0] == 1
    assert list(frame.columns) == ["a", "b"]


def test_pipeline_probabilities_are_bounded(sample_clean_data):
    X, y = split_features_target(sample_clean_data)
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
