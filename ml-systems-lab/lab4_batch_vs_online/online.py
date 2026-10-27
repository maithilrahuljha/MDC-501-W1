"""
WHAT: Serves the churn model one customer at a time over HTTP, with validation -- the ONLINE half of Lab 4.
HOW:  python online.py     (leave it running), then in a second terminal: python batch.py  and  python benchmark.py
OK:   prints "Uvicorn running on http://127.0.0.1:8000" and GET /health returns {"status":"ok"}.
"""

# =============================================================================
# BATCH vs ONLINE -- the same model, two doors
# -----------------------------------------------------------------------------
# BATCH  : "here are 10,000 rows, answer when you are done." One vectorised call.
#          Great for nightly scoring, reports, churn campaigns.
# ONLINE : "here is one row, answer in 50 milliseconds." One HTTP request each.
#          Necessary for a website, a call centre screen, a mobile app.
#
# The MODEL is identical. Only the SHAPE OF THE CALL changes. That is the
# lesson: batch vs online is a deployment decision, not a modelling decision.
#
# The model is shared with batch.py and benchmark.py through
# load_or_train_model(), so all three are guaranteed to use the exact same
# weights. If Lab 1 already produced a model, we reuse it.
# =============================================================================

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

# Lab 1 ships a trained model; reuse it if this lab is run from the full package.
LAB1_MODEL = BASE_DIR.parent / "lab1_break_notebook" / "artifacts" / "model.joblib"

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]


def make_training_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic churn data. Same generator as Lab 1, so the story stays consistent."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, size=n_rows)
    contract = rng.choice(CONTRACTS, size=n_rows, p=[0.5, 0.3, 0.2])
    charges = np.round(np.clip(rng.normal(65, 20, size=n_rows), 15, 160), 2)
    calls = rng.poisson(1.5, size=n_rows)

    contract_effect = np.where(contract == "monthly", 1.10,
                               np.where(contract == "one_year", 0.20, -0.90))
    logit = 1.6 * (
        0.045 * (30 - tenure)
        + 0.012 * (charges - 65)
        + 0.55 * calls
        + contract_effect
        - 0.40
    )
    churn = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_calls": calls,
            "contract_type": contract,
            "churn": churn,
        }
    )


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def load_or_train_model():
    """Return (model, metadata). Reuse Lab 1's model, or FileNotFound? train one now."""
    if LAB1_MODEL.exists():
        model = joblib.load(LAB1_MODEL)
        meta_path = LAB1_MODEL.parent / "model_metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return model, {**metadata, "source": "lab1_break_notebook/artifacts/model.joblib"}

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
        return model, {**metadata, "source": "lab4_batch_vs_online/artifacts/model.joblib"}

    print("[online] No saved model found. Training a fresh one (about 2 seconds) ...")
    data = make_training_data()
    model = build_pipeline().fit(data[FEATURES], data["churn"])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_version": "1.0.0-lab4",
        "trained_on_rows": int(len(data)),
        "source": "trained inside lab4_batch_vs_online/online.py",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[online] Model saved to {MODEL_PATH}")
    return model, metadata


MODEL, METADATA = load_or_train_model()

app = FastAPI(
    title="Lab 4 - Batch vs Online",
    description="The same churn model, scored one customer at a time.",
    version=str(METADATA.get("model_version", "unknown")),
)


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_calls: int = Field(ge=0, le=100)
    contract_type: Literal["monthly", "one_year", "two_year"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": METADATA.get("model_version", "unknown"),
        "model_source": METADATA.get("source", "unknown"),
    }


@app.post("/predict")
def predict(customer: Customer):
    # One row in, one answer out. The cost of this shape of call is per-request
    # overhead (HTTP parsing, JSON, a thread hop) -- not model arithmetic.
    row = pd.DataFrame([customer.model_dump()])
    probability = float(MODEL.predict_proba(row[FEATURES])[0, 1])
    return {
        "churn_probability": round(probability, 4),
        "label": int(probability >= 0.5),
        "model_version": METADATA.get("model_version", "unknown"),
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print()
    print("=" * 68)
    print("  Lab 4 online service")
    print(f"  Model        : {METADATA.get('source', 'unknown')}")
    print(f"  Endpoint     : http://127.0.0.1:{port}/predict   (POST, one customer)")
    print(f"  Health check : http://127.0.0.1:{port}/health")
    print("  Leave this window open. Use a second terminal for batch/benchmark.")
    print("=" * 68)
    print()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
