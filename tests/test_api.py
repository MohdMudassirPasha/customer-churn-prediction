from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src import predict as predict_module
from src.config import Settings
from tests.test_predict import VALID_CUSTOMER


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    temp_settings: Settings,
    fitted_model_artifact,
    model_info_file,
) -> TestClient:
    """A TestClient whose routes resolve to the temporary model artifacts."""
    import api.routes as routes

    monkeypatch.setattr(routes, "settings", temp_settings, raising=True)
    predict_module.clear_model_cache()

    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    predict_module.clear_model_cache()


def test_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health_ok(client: TestClient):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client: TestClient):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "Logistic Regression"


def test_predict_valid_payload(client: TestClient):
    response = client.post("/predict", json=VALID_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction_label"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_enum(client: TestClient):
    payload = {**VALID_CUSTOMER, "gender": "Other"}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_extra_fields(client: TestClient):
    payload = {**VALID_CUSTOMER, "unexpected_field": 1}
    assert client.post("/predict", json=payload).status_code == 422


def test_predict_defaults_total_charges(client: TestClient):
    payload = {key: value for key, value in VALID_CUSTOMER.items() if key != "TotalCharges"}
    assert client.post("/predict", json=payload).status_code == 200


def test_openapi_schema_served(client: TestClient):
    assert client.get("/openapi.json").status_code == 200


def test_startup_warms_model_cache(client: TestClient):
    # Entering the TestClient context manager runs the lifespan handler, which
    # should have populated the model cache before any request is served.
    assert predict_module.load_model_artifact.cache_info().currsize >= 1
