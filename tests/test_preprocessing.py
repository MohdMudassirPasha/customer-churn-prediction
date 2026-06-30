from __future__ import annotations

import numpy as np

from src.feature_engineering import CustomerChurnFeatureEngineer, tenure_group
from src.pipeline import build_pipeline
from src.preprocessing import clean_telco_data, split_features_target


def test_clean_telco_data_normalizes_target_and_numeric_columns(sample_raw_data):
    cleaned = clean_telco_data(sample_raw_data)

    assert cleaned["Churn"].isin([0, 1]).all()
    assert cleaned["TotalCharges"].dtype.kind in {"f", "i"}
    assert cleaned["MonthlyCharges"].dtype.kind in {"f", "i"}
    assert cleaned["tenure"].dtype.kind in {"f", "i"}


def test_split_features_target_drops_id_and_target(sample_clean_data):
    X, y = split_features_target(sample_clean_data)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}


def test_feature_engineering_adds_expected_columns(sample_clean_data):
    X, _ = split_features_target(sample_clean_data)
    transformed = CustomerChurnFeatureEngineer().fit_transform(X)

    assert "TenureGroup" in transformed.columns
    assert "ServiceCount" in transformed.columns
    assert "IsMonthToMonth" in transformed.columns
    assert "UsesElectronicCheck" in transformed.columns
    assert tenure_group(72) == "61+"


def test_pipeline_can_fit_and_score(sample_clean_data):
    X, y = split_features_target(sample_clean_data)
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X.head(5))[:, 1]
    assert probabilities.shape == (5,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
