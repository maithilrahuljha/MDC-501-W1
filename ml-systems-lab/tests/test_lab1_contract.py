"""
WHAT: Proves Lab 1's lesson with automated tests -- one good request, three bad ones.
HOW:  from the ml-systems-lab folder:  pytest tests -v
OK:   all tests pass; the validated endpoint answers 200 once and 422 three times.
"""

import json
import shutil

import pytest
from conftest import LAB1, load_module
from fastapi.testclient import TestClient

GOOD = {
    "tenure_months": 12,
    "monthly_charges": 79.5,
    "support_calls": 2,
    "contract_type": "monthly",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Train a throwaway model, then import the app against it."""
    import joblib

    train = load_module(LAB1 / "train.py", "lab1_train")

    artifacts = tmp_path_factory.mktemp("artifacts")
    df = train.make_synthetic_churn_data(n_rows=600, seed=1)
    model = train.build_pipeline().fit(
        df[train.NUMERIC_FEATURES + train.CATEGORICAL_FEATURES], df[train.TARGET]
    )
    joblib.dump(model, artifacts / "model.joblib")
    (artifacts / "model_metadata.json").write_text(
        json.dumps({"model_version": "test", "trained_on_rows": 600, "accuracy": 0.8}),
        encoding="utf-8",
    )

    lab1_app = load_module(LAB1 / "app.py", "lab1_app")

    lab1_app.model = model
    lab1_app.metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    with TestClient(lab1_app.app, raise_server_exceptions=False) as test_client:
        yield test_client
    shutil.rmtree(artifacts, ignore_errors=True)


def test_health_reports_a_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_valid_request_is_scored(client):
    response = client.post("/predict_validated", json=GOOD)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["validated"] is True


def test_missing_column_is_rejected(client):
    """(a) MISSING COLUMN. Pillar 1: data validation."""
    body = {k: v for k, v in GOOD.items() if k != "monthly_charges"}
    assert client.post("/predict", json=body).status_code == 500          # the naive way
    assert client.post("/predict_validated", json=body).status_code == 422  # the fixed way


def test_wrong_type_is_rejected(client):
    """(b) WRONG DATA TYPE. Pillar 4: serving contract."""
    body = {**GOOD, "tenure_months": "twelve"}
    assert client.post("/predict", json=body).status_code == 500
    assert client.post("/predict_validated", json=body).status_code == 422


def test_unseen_category_is_rejected(client):
    """(c) UNSEEN CATEGORY. Pillar 1 (block it) + Pillar 5 (notice it happened)."""
    body = {**GOOD, "contract_type": "lifetime"}
    assert client.post("/predict", json=body).status_code == 500
    assert client.post("/predict_validated", json=body).status_code == 422
