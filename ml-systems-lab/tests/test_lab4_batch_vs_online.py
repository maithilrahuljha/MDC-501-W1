"""
WHAT: Checks that batch and online scoring use the same model and give the SAME answers for the same row.
HOW:  from the ml-systems-lab folder:  pytest tests -v
OK:   all tests pass; batch and online probabilities match to within rounding.
"""

from pathlib import Path

import pandas as pd
import pytest
from conftest import load_module
from fastapi.testclient import TestClient

lab4_online = load_module(Path(__file__).resolve().parent.parent / "lab4_batch_vs_online" / "online.py",
                         "lab4_online")
lab4_batch = load_module(Path(__file__).resolve().parent.parent / "lab4_batch_vs_online" / "batch.py",
                         "lab4_batch")


@pytest.fixture(scope="module")
def client():
    with TestClient(lab4_online.app) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_source"]


def test_online_rejects_bad_contract(client):
    response = client.post(
        "/predict",
        json={"tenure_months": 12, "monthly_charges": 79.5, "support_calls": 2,
              "contract_type": "lifetime"},
    )
    assert response.status_code == 422


def test_online_and_batch_agree(client):
    """THE point of Lab 4: one model, two doors, same answer."""
    rows = lab4_batch.generate_input_rows(25, seed=99)
    batch_predictions, _ = lab4_batch.score_batch(lab4_online.MODEL, rows)

    for index, row in rows.head(5).iterrows():
        payload = {key: row[key] for key in lab4_online.FEATURES}
        payload["tenure_months"] = int(payload["tenure_months"])
        payload["support_calls"] = int(payload["support_calls"])
        payload["monthly_charges"] = float(payload["monthly_charges"])
        online_probability = client.post("/predict", json=payload).json()["churn_probability"]
        assert online_probability == pytest.approx(batch_predictions.iloc[index]["churn_probability"], abs=1e-4)


def test_batch_input_file_can_be_recreated(tmp_path):
    path = tmp_path / "rows.csv"
    lab4_batch.ensure_input_file(50, path)
    assert path.exists()
    frame = pd.read_csv(path)
    assert len(frame) == 50
    assert list(frame.columns) == lab4_online.FEATURES
