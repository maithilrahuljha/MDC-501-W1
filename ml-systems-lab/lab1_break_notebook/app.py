"""
WHAT: A deliberately fragile FastAPI service that trusts every request, so Lab 1 can break it in public.
HOW:  python app.py   (leave it running), then in a second terminal run: python test_requests.py
OK:   GET /health returns {"status":"ok"} with an HTTP 200, and POST /predict returns a churn probability.
"""

# =============================================================================
# READ ME FIRST -- the whole point of this file
# =============================================================================
# /predict          -> the NAIVE endpoint. It does no checking. It crashes.
# /predict_validated-> the same model, but behind a contract. It says "no" nicely.
#
# The three failures we want you to see, and the FIVE PILLARS that would have
# prevented each one:
#
#   (a) MISSING COLUMN   -> prevented by Pillar 1: DATA VALIDATION.
#                           The service should reject a body that is missing a
#                           field BEFORE the model is asked to score it.
#   (b) WRONG DATA TYPE  -> prevented by Pillar 4: SERVING CONTRACT.
#                           "12" and 12 are not the same thing. The API should
#                           publish and enforce a schema (see /predict_validated).
#   (c) UNSEEN CATEGORY  -> prevented by Pillar 1 (DATA VALIDATION) for the
#                           incoming value, and Pillar 5 (MONITORING) to notice
#                           that a brand-new category appeared in the real world.
#
# Plus: every crash below is Pillar 2 (TESTING) territory, because a single
# automated test would have caught it before a user did.
# =============================================================================

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

ARTIFACT_HELP = (
    "\n[app.py] I cannot find artifacts/model.joblib.\n"
    "         Fix: run 'python train.py' first (in this same folder),\n"
    "         then run 'python app.py' again.\n"
)

# NOTE (this line is itself a lesson): we load the model WITHOUT crashing at
# import time. Two reasons:
#   1. Importing a module must not kill the process. A test runner, a container,
#      or `uvicorn app:app` should be able to read this file with no model on disk.
#   2. A missing model is a 503 "not ready", not a dead server. A crash takes the
#      whole process down; a 503 keeps the health check answering while someone
#      fixes it.
# Students still get the friendly message, because __main__ checks before serving.
# (We learned this the hard way: our own CI went red, because a fresh clone has no
# artifacts/ folder. The tests were quietly relying on a file on the developer's
# laptop. Pillar 2 catches exactly this kind of thing.)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
ALLOWED_CONTRACT_TYPES = metadata.get("allowed_categories", {}).get(
    "contract_type", ["monthly", "one_year", "two_year"]
)

app = FastAPI(
    title="Lab 1 - Break the Notebook",
    description="A model in a notebook is not a production system. This is why.",
    version=metadata.get("model_version", "0.0.0"),
)

STATS = {"requests": 0, "errors": 0}


# -----------------------------------------------------------------------------
# A dev-mode crash handler. On purpose, it returns the raw error to the caller
# and prints a loud banner in the server terminal. Never do this in real life:
# error strings leak inside information. We do it here so you can SEE failure.
# -----------------------------------------------------------------------------
def loud_crash(path: str, exc: Exception) -> JSONResponse:
    """Print a loud banner in the server terminal and return a deliberate 500."""
    STATS["errors"] += 1
    print()
    print("!" * 72, flush=True)
    print(f"!! UNHANDLED CRASH on POST {path}", flush=True)
    print(f"!! Error type : {type(exc).__name__}", flush=True)
    print(f"!! Message    : {str(exc)[:400]}", flush=True)
    print("!! The model was asked to score something it cannot understand.", flush=True)
    print("!! Which pillar would have prevented this? See the notes at the top of app.py.", flush=True)
    print("!" * 72, flush=True)
    print()
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc)[:400],
            "note": "This is a deliberate 500. The naive endpoint validates nothing.",
        },
    )


def model_not_ready() -> JSONResponse:
    """503: the service is up, but it has nothing to predict with. Honest, not dead."""
    return JSONResponse(status_code=503, content={"error": "model_not_loaded",
                                                  "detail": ARTIFACT_HELP.strip()})


# -----------------------------------------------------------------------------
# (1) NAIVE ENDPOINT - exactly what a notebook export looks like.
#     No schema, no checks, no limits. Any dict is accepted and shoved at the model.
# -----------------------------------------------------------------------------
@app.post("/predict")
def predict_naive(payload: dict):
    STATS["requests"] += 1
    if model is None:
        return model_not_ready()
    try:
        row = pd.DataFrame([payload])      # hope it is right. it is not.
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:               # noqa: BLE001 - catching on purpose
        # A real service would log this and return a generic 500. We are noisy
        # on purpose so the failure is impossible to miss during the lab.
        return loud_crash("/predict", exc)
    return {
        "churn_probability": round(probability, 4),
        "label": int(probability >= 0.5),
        "model_version": metadata.get("model_version", "unknown"),
    }


# -----------------------------------------------------------------------------
# (2) THE FIX - same model, but the door has a frame around it.
#     Pydantic enforces: every field present, right type, category from a list.
#     FastAPI turns every violation into a readable HTTP 422 before the model
#     is ever called. This is Pillars 1 + 4 working together.
# -----------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120, description="Months as a customer")
    monthly_charges: float = Field(ge=0, le=1000, description="Monthly bill in USD")
    support_calls: int = Field(ge=0, le=100, description="Support calls last month")
    contract_type: Literal["monthly", "one_year", "two_year"] = Field(
        description="Contract type. Anything else is rejected."
    )


@app.post("/predict_validated")
def predict_validated(request: ChurnRequest):
    STATS["requests"] += 1
    if model is None:
        return model_not_ready()
    try:
        row = pd.DataFrame([request.model_dump()])
        probability = float(model.predict_proba(row)[0, 1])
    except Exception as exc:               # noqa: BLE001
        return loud_crash("/predict_validated", exc)
    return {
        "churn_probability": round(probability, 4),
        "label": int(probability >= 0.5),
        "model_version": metadata.get("model_version", "unknown"),
        "validated": True,
    }


@app.get("/health")
def health():
    # Pillar 5 (MONITORING) starts here: something must be able to ask
    # "are you alive, and which model are you serving?" at 3 a.m.
    return {
        "status": "ok",                       # the process is alive
        "model_loaded": model is not None,    # ...but is it useful yet?
        "model_version": metadata.get("model_version", "unknown"),
        "trained_on_rows": metadata.get("trained_on_rows"),
        "offline_accuracy": metadata.get("accuracy"),
    }


@app.get("/stats")
def stats():
    # A baby step toward observability (Pillar 5). Real services export metrics.
    return {**STATS, "allowed_contract_types": ALLOWED_CONTRACT_TYPES}


if __name__ == "__main__":
    if model is None:          # `python app.py` should still explain itself politely
        raise SystemExit(ARTIFACT_HELP)
    port = int(os.environ.get("PORT", "8000"))
    print()
    print("=" * 68)
    print("  Lab 1 service starting")
    print(f"  Model version : {metadata.get('model_version', 'unknown')}")
    print(f"  Docs page     : http://127.0.0.1:{port}/docs")
    print(f"  Health check  : http://127.0.0.1:{port}/health")
    print("  Leave this window open. Open a SECOND terminal for test_requests.py")
    print("=" * 68)
    print()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
