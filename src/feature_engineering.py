"""Feature engineering transformers for the Telco churn dataset."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

YES_NO_COLUMNS = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def normalize_binary(value: Any) -> int:
    """Normalize Yes/No-like values to 1/0 while treating unknowns as 0."""
    if pd.isna(value):
        return 0
    normalized = str(value).strip().lower()
    return int(normalized in {"yes", "true", "1", "y"})


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric values and keep invalid values as NaN."""
    return pd.to_numeric(series.replace(" ", np.nan), errors="coerce")


def tenure_group(tenure: float | int | None) -> str:
    """Convert tenure months into stable business buckets."""
    if pd.isna(tenure):
        return "unknown"
    value = float(tenure)
    if value <= 12:
        return "0-12"
    if value <= 24:
        return "13-24"
    if value <= 48:
        return "25-48"
    if value <= 60:
        return "49-60"
    return "61+"


class CustomerChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain features while preserving the dataframe contract."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> CustomerChurnFeatureEngineer:
        """Return the fitted transformer."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe with engineered churn features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        data = X.copy()
        for column in data.select_dtypes(include=["object", "string"]).columns:
            data[column] = data[column].astype("string").str.strip()

        if "TotalCharges" in data.columns:
            data["TotalCharges"] = safe_numeric(data["TotalCharges"])
        if "MonthlyCharges" in data.columns:
            data["MonthlyCharges"] = safe_numeric(data["MonthlyCharges"])
        if "tenure" in data.columns:
            data["tenure"] = safe_numeric(data["tenure"])
        if "SeniorCitizen" in data.columns:
            data["SeniorCitizen"] = safe_numeric(data["SeniorCitizen"]).fillna(0).astype(int)

        data = self._add_charge_features(data)
        data = self._add_tenure_features(data)
        data = self._add_service_features(data)
        data = self._add_contract_payment_features(data)
        return data

    @staticmethod
    def _add_charge_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add charge-based interaction features."""
        if {"MonthlyCharges", "tenure"}.issubset(data.columns):
            tenure_nonzero = data["tenure"].replace(0, np.nan)
            data["AvgChargesPerTenure"] = (data["MonthlyCharges"] / tenure_nonzero).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if {"TotalCharges", "MonthlyCharges"}.issubset(data.columns):
            monthly_nonzero = data["MonthlyCharges"].replace(0, np.nan)
            data["TotalToMonthlyRatio"] = (data["TotalCharges"] / monthly_nonzero).replace(
                [np.inf, -np.inf],
                np.nan,
            )
        if "MonthlyCharges" in data.columns:
            data["HighMonthlyCharges"] = (data["MonthlyCharges"] >= 80).astype(int)
        return data

    @staticmethod
    def _add_tenure_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add tenure bucket and early-life indicators."""
        if "tenure" in data.columns:
            data["TenureGroup"] = data["tenure"].apply(tenure_group)
            data["IsNewCustomer"] = (data["tenure"].fillna(0) <= 6).astype(int)
            data["IsLongTermCustomer"] = (data["tenure"].fillna(0) >= 48).astype(int)
        return data

    @staticmethod
    def _add_service_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add service adoption features."""
        available_services = [column for column in SERVICE_COLUMNS if column in data.columns]
        if available_services:
            data["ServiceCount"] = data[available_services].apply(
                lambda row: sum(normalize_binary(value) for value in row),
                axis=1,
            )
            data["HasAnyProtection"] = data[
                [column for column in ["OnlineSecurity", "TechSupport"] if column in data.columns]
            ].apply(lambda row: int(any(normalize_binary(value) for value in row)), axis=1)
            data["HasStreaming"] = data[
                [column for column in ["StreamingTV", "StreamingMovies"] if column in data.columns]
            ].apply(lambda row: int(any(normalize_binary(value) for value in row)), axis=1)
        if "InternetService" in data.columns:
            data["HasInternetService"] = (
                data["InternetService"].fillna("No").astype(str).str.lower() != "no"
            ).astype(int)
        return data

    @staticmethod
    def _add_contract_payment_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add contract and payment behavior features."""
        if "Contract" in data.columns:
            contract = data["Contract"].fillna("").astype(str).str.lower()
            data["IsMonthToMonth"] = contract.eq("month-to-month").astype(int)
            data["HasLongTermContract"] = contract.isin(["one year", "two year"]).astype(int)
        if "PaymentMethod" in data.columns:
            payment = data["PaymentMethod"].fillna("").astype(str).str.lower()
            data["UsesElectronicCheck"] = payment.str.contains(
                "electronic check", regex=False
            ).astype(int)
            data["UsesAutomaticPayment"] = payment.str.contains(
                "bank transfer|credit card",
                regex=True,
            ).astype(int)
        return data
