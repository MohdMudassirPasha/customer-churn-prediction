"""Data cleaning and preprocessing utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import settings


def clean_telco_data(
    data: pd.DataFrame, target_column: str = settings.target_column
) -> pd.DataFrame:
    """Clean raw Telco data without fitting statistical transforms."""
    cleaned = data.copy()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].astype("string").str.strip()

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(
            cleaned["TotalCharges"].replace(" ", np.nan),
            errors="coerce",
        )
    if "MonthlyCharges" in cleaned.columns:
        cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    if "tenure" in cleaned.columns:
        cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")

    if target_column in cleaned.columns:
        cleaned = cleaned[cleaned[target_column].notna()].copy()
        cleaned[target_column] = cleaned[target_column].map(
            {
                settings.positive_label: 1,
                settings.negative_label: 0,
                1: 1,
                0: 0,
                "1": 1,
                "0": 0,
            }
        )
        cleaned = cleaned[cleaned[target_column].notna()].copy()
        cleaned[target_column] = cleaned[target_column].astype(int)

    return cleaned


def split_features_target(
    data: pd.DataFrame,
    target_column: str = settings.target_column,
    id_column: str = settings.id_column,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned dataframe into features and target."""
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' is missing.")

    y = data[target_column].astype(int)
    drop_columns = [target_column]
    if id_column in data.columns:
        drop_columns.append(id_column)
    X = data.drop(columns=drop_columns)
    return X, y


def infer_column_types(
    data: pd.DataFrame,
    excluded_columns: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns."""
    excluded = set(excluded_columns or [])
    feature_data = data.drop(columns=[column for column in excluded if column in data.columns])
    numeric_features = feature_data.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [
        column for column in feature_data.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


class DataFramePreprocessor(BaseEstimator, TransformerMixin):
    """Fit a dataframe-aware preprocessing graph for mixed feature types."""

    def __init__(self) -> None:
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []
        self.transformer_: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DataFramePreprocessor:
        """Fit imputers, scaler, and encoder."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.numeric_features_, self.categorical_features_ = infer_column_types(X)
        transformers: list[tuple[str, Pipeline, list[str]]] = []

        if self.numeric_features_:
            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("numeric", numeric_pipeline, self.numeric_features_))

        if self.categorical_features_:
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("categorical", categorical_pipeline, self.categorical_features_))

        if not transformers:
            raise ValueError("No usable feature columns found for preprocessing.")

        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        self.transformer_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform a dataframe into a numeric matrix."""
        if self.transformer_ is None:
            raise RuntimeError("DataFramePreprocessor must be fitted before transform.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return self.transformer_.transform(X)

    def get_feature_names_out(self) -> np.ndarray:
        """Return transformed feature names."""
        if self.transformer_ is None:
            raise RuntimeError(
                "DataFramePreprocessor must be fitted before feature names are available."
            )
        return self.transformer_.get_feature_names_out()
