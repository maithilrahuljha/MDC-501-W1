"""
WHAT: Trains a small churn model on synthetic data and saves artifacts/model.joblib + artifacts/model_metadata.json.
HOW:  cd lab1_break_notebook && python train.py
OK:   prints "TRAINING COMPLETE" plus a test accuracy between about 0.78 and 0.95, and creates the artifacts/ folder.
"""

# ---------------------------------------------------------------------------
# Lab 1, step 1: the "notebook" part. This is the easy, happy-path code that
# everybody writes first. It works perfectly on the data it was trained on.
# Nothing here is wrong. Everything here is *incomplete* for production.
# ---------------------------------------------------------------------------

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
N_ROWS = 4000
MODEL_VERSION = "1.0.0"

# These are the five pillars we use in every lab in this course.
# (They are deliberately simple names so you can remember them.)
#   1. DATA VALIDATION   - check what arrives before the model touches it.
#   2. TESTING           - prove the thing works, automatically, every time.
#   3. VERSIONING        - know which data + code + model produced this answer.
#   4. SERVING           - a contract, a schema, and a latency budget.
#   5. MONITORING        - log what happens in the wild and alert when it drifts.

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL_FEATURES = ["contract_type"]
TARGET = "churn"
ALLOWED_CONTRACT_TYPES = ["monthly", "one_year", "two_year"]

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"


def make_synthetic_churn_data(n_rows: int = N_ROWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Build a fake-but-believable telecom churn table. No download, no login."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, size=n_rows)                      # months with us
    contract = rng.choice(ALLOWED_CONTRACT_TYPES, size=n_rows, p=[0.5, 0.3, 0.2])
    charges = np.round(np.clip(rng.normal(65, 20, size=n_rows), 15, 160), 2)
    calls = rng.poisson(1.5, size=n_rows)                           # support calls

    contract_effect = np.where(contract == "monthly", 1.10,
                               np.where(contract == "one_year", 0.20, -0.90))
    signal = 1.6  # just a volume knob so the task is learnable but not trivial
    logit = signal * (
        0.045 * (30 - tenure)      # newer customers churn more
        + 0.012 * (charges - 65)   # price hurts
        + 0.55 * calls             # unhappy customers churn
        + contract_effect          # long contracts keep people
        - 0.40                     # intercept, keeps churn rate sane
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(n_rows) < probability).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_calls": calls,
            "contract_type": contract,
            TARGET: churn,
        }
    )


def build_pipeline() -> Pipeline:
    """One object that does preprocessing AND prediction. Saved as one file."""
    numeric = Pipeline(steps=[("scale", StandardScaler())])
    # NOTE: OneHotEncoder's default is handle_unknown="error", so a category the
    # model has never seen becomes a hard crash later on. We keep the default on
    # purpose: in Lab 1 we WANT the crash to be visible. (Pillar 1 - data
    # validation - is what should stop bad categories long before this point.)
    categorical = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False))]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Generating synthetic churn data ...")
    df = make_synthetic_churn_data()
    csv_path = ARTIFACT_DIR / "training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"      {len(df):,} rows written to {csv_path.relative_to(BASE_DIR)}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    print("[2/4] Training the pipeline ...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("[3/4] Evaluating on the held-out 25% ...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print("[4/4] Saving artifacts ...")
    model_path = ARTIFACT_DIR / "model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_on_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": round(float(accuracy), 4),
        "roc_auc": round(float(auc), 4),
        "required_columns": {
            "tenure_months": "int",
            "monthly_charges": "float",
            "support_calls": "int",
            "contract_type": "string",
        },
        "allowed_categories": {"contract_type": ALLOWED_CONTRACT_TYPES},
        "positive_class": "churn = 1 (customer leaves)",
    }
    metadata_path = ARTIFACT_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("=" * 68)
    print("TRAINING COMPLETE")
    print(f"  test accuracy : {accuracy:.3f}")
    print(f"  test ROC-AUC  : {auc:.3f}")
    print(f"  model file    : {model_path.relative_to(BASE_DIR)}")
    print(f"  metadata file : {metadata_path.relative_to(BASE_DIR)}")
    print("=" * 68)
    print()
    print("A valid request body looks like this:")
    print(json.dumps(
        {
            "tenure_months": 12,
            "monthly_charges": 79.5,
            "support_calls": 2,
            "contract_type": "monthly",
        },
        indent=2,
    ))
    print()
    print("Next step: run 'python app.py' and leave it running, then open a")
    print("SECOND terminal and run 'python test_requests.py'.")


if __name__ == "__main__":
    main()
