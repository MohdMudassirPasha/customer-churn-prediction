"""Pydantic schemas for the churn prediction API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]


class TelcoCustomer(BaseModel):
    """Validated IBM Telco customer payload."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        },
    )

    customer_id: str | None = Field(default=None, alias="customerID")
    gender: Literal["Female", "Male"]
    senior_citizen: int = Field(alias="SeniorCitizen", ge=0, le=1)
    partner: YesNo = Field(alias="Partner")
    dependents: YesNo = Field(alias="Dependents")
    tenure: int = Field(ge=0, le=120)
    phone_service: YesNo = Field(alias="PhoneService")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(alias="MultipleLines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="InternetService")
    online_security: Literal["Yes", "No", "No internet service"] = Field(alias="OnlineSecurity")
    online_backup: Literal["Yes", "No", "No internet service"] = Field(alias="OnlineBackup")
    device_protection: Literal["Yes", "No", "No internet service"] = Field(alias="DeviceProtection")
    tech_support: Literal["Yes", "No", "No internet service"] = Field(alias="TechSupport")
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(alias="StreamingTV")
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(alias="StreamingMovies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: YesNo = Field(alias="PaperlessBilling")
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(alias="PaymentMethod")
    monthly_charges: float = Field(alias="MonthlyCharges", ge=0)
    total_charges: float | None = Field(default=None, alias="TotalCharges", ge=0)

    @model_validator(mode="after")
    def default_total_charges(self) -> TelcoCustomer:
        """Use monthly charges times tenure when total charges is omitted."""
        if self.total_charges is None:
            self.total_charges = float(self.monthly_charges) * float(self.tenure)
        return self

    def to_model_record(self) -> dict[str, Any]:
        """Return a model-ready dictionary using IBM column names."""
        if hasattr(self, "model_dump"):
            return self.model_dump(by_alias=True, exclude_none=True)
        return self.dict(by_alias=True, exclude_none=True)


class PredictionResponse(BaseModel):
    """Prediction response payload."""

    prediction: int
    prediction_label: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    model_name: str


class HealthResponse(BaseModel):
    """Service health response."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_name: str
    artifact_path: str
    decision_threshold: float
    metrics: dict[str, float]
    feature_count: int
    features: list[str]
    trained_at: str | None = None
    optuna_best_params: dict[str, Any] | None = None
