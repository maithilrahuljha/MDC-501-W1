"""
WHAT: Trains the churn model as a tracked pipeline run -- locally with MLflow (--mode local) or on Vertex AI (--mode vertex).
HOW:  python pipeline.py --mode local        |   python pipeline.py --mode vertex --project my-project --bucket gs://my-bucket
OK:   local mode prints an MLflow run id and metrics; vertex mode prints a submitted job or a friendly "no credentials" message.
"""

# =============================================================================
# WHAT THIS FILE TEACHES
# -----------------------------------------------------------------------------
# A "pipeline" is just the steps of your notebook written down as a program:
#   make data -> split -> train -> evaluate -> save/log the model.
# Running it as a SCRIPT is what makes it reproducible and schedulable.
#
# Two backends, one script:
#
#   --mode local   MLflow (open source, runs on your laptop, free)
#                  Tracks params, metrics and model versions in ./mlruns
#
#   --mode vertex  Vertex AI (Google Cloud, paid, managed)
#                  Submits the same training as a cloud job with the same
#                  artifacts, but you need a Google Cloud project + credentials.
#
# The mapping is nearly one-to-one. If you can read this file, you can move
# between a free local setup and a managed cloud one without learning a new
# mental model:
#
#   mlflow.start_run()                 <->  aiplatform.CustomTrainingJob(...).run()
#   mlflow.log_param / log_metric      <->  job metrics in the Vertex console
#   mlflow.sklearn.log_model(...)      <->  aiplatform.Model.upload(...)
#   registered model "churn"           <->  Vertex Model Registry
#   run id                             <->  job resource name
#
# Nothing here needs a credit card in --mode local. Vertex mode is OPTIONAL.
# =============================================================================

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]
EXPERIMENT_NAME = "mdc501-churn"
REGISTERED_MODEL_NAME = "churn-classifier"
REGION = "us-central1"
SKLEARN_TRAIN_IMAGE = "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest"


# =============================================================================
# The pipeline steps. Identical for both backends -- that is the point.
# =============================================================================
def make_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic churn data. No download, no login, no secrets."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 72, size=n_rows)
    contract = rng.choice(CONTRACTS, size=n_rows, p=[0.5, 0.3, 0.2])
    charges = np.round(np.clip(rng.normal(65, 20, size=n_rows), 15, 160), 2)
    calls = rng.poisson(1.5, size=n_rows)

    contract_effect = np.where(contract == "monthly", 1.10,
                               np.where(contract == "one_year", 0.20, -0.90))
    logit = 1.6 * (
        0.045 * (30 - tenure) + 0.012 * (charges - 65) + 0.55 * calls
        + contract_effect - 0.40
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


def train(data: pd.DataFrame, seed: int = 42, c_value: float = 1.0):
    """Split, fit, evaluate. Returns (model, metrics, params)."""
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["churn"], test_size=0.25, random_state=seed, stratify=data["churn"]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, C=c_value, random_state=seed)),
        ]
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "f1_churn": round(float(f1_score(y_test, predictions)), 4),
        "log_loss": round(float(log_loss(y_test, probabilities)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    params = {"model_type": "logistic_regression", "C": c_value, "seed": seed,
              "n_rows": int(len(data)), "features": ",".join(FEATURES)}
    return model, metrics, params


def print_metrics(metrics: dict) -> None:
    print("\n  Metrics on the held-out 25%:")
    for key, value in metrics.items():
        print(f"    {key:<12} {value}")


# =============================================================================
# BACKEND 1 -- MLflow, on your own computer, free.
# Vertex AI equivalent of everything in this function is noted in run_vertex().
# =============================================================================
def run_local(args) -> int:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("[local] MLflow is not installed, so I cannot track this run.")
        print("        Fix: pip install mlflow==2.17.2")
        print("        (Or run with --mode vertex-train to just train and save a file.)")
        return 0

    tracking_uri = args.tracking_uri or (Path(args.model_dir).parent / "mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"[local] Tracking server : {tracking_uri}")
    print(f"[local] Experiment      : {EXPERIMENT_NAME}")

    data = make_data(args.rows, args.seed)
    model, metrics, params = train(data, args.seed, args.c_value)

    print(f"[local] Trained on {len(data):,} rows, {len(FEATURES)} features")
    print_metrics(metrics)

    with mlflow.start_run(run_name=f"logreg-c{args.c_value}-seed{args.seed}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        # Save the exact data snapshot too: a model without its data is folklore.
        snapshot = Path(tempfile.mkdtemp()) / "training_snapshot.csv"
        data.head(500).to_csv(snapshot, index=False)
        mlflow.log_artifact(str(snapshot), artifact_path="data")

        # Vertex equivalent: aiplatform.Model.upload(display_name=..., artifact_uri=...)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print()
        print("=" * 70)
        print("MLFLOW RUN COMPLETE")
        print("=" * 70)
        print(f"  run id            : {run.info.run_id}")
        print(f"  registered model  : {REGISTERED_MODEL_NAME} (versioned automatically)")
        print(f"  artifacts live in : {tracking_uri}")
        print()
        print("  See it with your own eyes:")
        print(f"    mlflow ui --backend-store-uri \"{tracking_uri}\"")
        print("  then open the address it prints (usually http://127.0.0.1:5000).")
        print("  Compare two runs by changing --c-value and running again.")
        print("=" * 70)
        print()
        print("  OPEN-SOURCE EQUIVALENT of every Vertex AI concept in this lab:")
        print("    Vertex Experiment   ->  MLflow experiment")
        print("    Vertex Run          ->  MLflow run")
        print("    Vertex Model Reg.   ->  MLflow registered model")
        print("    Vertex Endpoint     ->  your own FastAPI service (Lab 1 and Lab 4)")
        print("  Same ideas. One costs money and one costs electricity.")
    return 0


# =============================================================================
# BACKEND 2 -- Vertex AI on Google Cloud. Needs a project, a bucket, credentials.
# Every call below has its MLflow twin written next to it.
# =============================================================================
def run_vertex(args) -> int:
    project = args.project or os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT")
    bucket = args.bucket or os.environ.get("GCS_BUCKET")

    print("[vertex] Checking what we have ...")
    print(f"         project : {project or '(not set)'}")
    print(f"         bucket  : {bucket or '(not set)'}")
    print(f"         region  : {args.region}")

    missing = []
    if not project:
        missing.append("a Google Cloud project id (--project or GOOGLE_CLOUD_PROJECT)")
    if not bucket:
        missing.append("a Cloud Storage bucket (--bucket or GCS_BUCKET), e.g. gs://my-churn-bucket")

    credentials_ok = True
    credentials_detail = ""
    try:
        import google.auth

        credentials, detected_project = google.auth.default()
        credentials_detail = f"credentials found (project detected: {detected_project or 'none'})"
        if not project and detected_project:
            project = detected_project
            missing = [m for m in missing if "project id" not in m]
    except ImportError:
        credentials_ok = False
        credentials_detail = "google-auth is not installed"
    except Exception as exc:  # DefaultCredentialsError and friends
        credentials_ok = False
        credentials_detail = f"{type(exc).__name__}: {str(exc)[:120]}"

    print(f"         auth    : {credentials_detail}")

    if missing or not credentials_ok:
        print()
        print("=" * 70)
        print("VERTEX MODE NEEDS A FEW THINGS FIRST -- nothing was sent to Google.")
        print("=" * 70)
        if missing:
            print("  Missing:")
            for item in missing:
                print(f"    - {item}")
        if not credentials_ok:
            print("  Missing: Google Cloud credentials on this computer.")
        print()
        print("  Fix it like this (one time, about 10 minutes):")
        print("    1. Install the Google Cloud CLI from https://cloud.google.com/sdk/docs/install")
        print("    2. Run:  gcloud auth application-default login")
        print("       (a browser opens; sign in with your Google account)")
        print("    3. Run:  gcloud config set project YOUR_PROJECT_ID")
        print("    4. Create a bucket:  gs://YOUR_BUCKET_NAME  (Console -> Cloud Storage)")
        print("    5. Run this script again with:")
        print("       python pipeline.py --mode vertex --project YOUR_PROJECT_ID "
              "--bucket gs://YOUR_BUCKET_NAME")
        print()
        print("  No Google Cloud account? That is completely fine.")
        print("  Use  --mode local  instead. It teaches the identical ideas for free,")
        print("  and the mapping between the two is written in the comments above.")
        print("=" * 70)
        return 0  # exit cleanly: this is an expected, handled situation

    try:
        from google.cloud import aiplatform
    except ImportError:
        print()
        print("[vertex] The Vertex AI library is not installed. Nothing was sent to Google.")
        print("         Fix: pip install google-cloud-aiplatform==1.71.1")
        print("         Or just use --mode local, which needs nothing extra.")
        return 0

    print("\n[vertex] Submitting the training job ...")
    # MLflow twin: mlflow.set_tracking_uri(...) / mlflow.set_experiment(...)
    aiplatform.init(project=project, location=args.region, staging_bucket=bucket)

    # MLflow twin: mlflow.sklearn.log_model(...) with a run started around it.
    # Here the SAME file you are reading is uploaded and run inside Google's
    # container. That is why it can be started with --mode vertex-train.
    job = aiplatform.CustomTrainingJob(
        display_name="churn-train",
        script_path=str(Path(__file__).resolve()),
        container_uri=SKLEARN_TRAIN_IMAGE,
        requirements=["scikit-learn==1.5.2", "pandas==2.2.3", "numpy==2.1.3", "joblib==1.4.2"],
        model_serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        ),
    )

    model = job.run(
        args=["--mode", "vertex-train", "--rows", str(args.rows), "--seed", str(args.seed)],
        replica_count=1,
        machine_type="n1-standard-4",
        model_display_name=REGISTERED_MODEL_NAME,
        sync=True,
        # MLflow twin: the run finishes and you read metrics from the run.
    )

    print()
    print("=" * 70)
    print("VERTEX AI JOB SUBMITTED AND FINISHED")
    print("=" * 70)
    print(f"  model resource : {model.resource_name}")
    print(f"  display name   : {REGISTERED_MODEL_NAME}")
    print("  Console path   : Vertex AI -> Training -> Custom jobs")
    print()
    print("  Next, in real life, you would deploy this model to an Endpoint")
    print("  (aiplatform.Endpoint) and pay per hour it stays up. For this")
    print("  course, stop here: the point was to see that the SAME pipeline")
    print("  code can run on your laptop or in a cloud job.")
    print("=" * 70)
    return 0


# =============================================================================
# ENTRYPOINT USED INSIDE THE CLOUD CONTAINER (and testable locally).
# Vertex sets AIP_MODEL_DIR; we must save our model there.
# =============================================================================
def run_vertex_train(args) -> int:
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    data = make_data(args.rows, args.seed)
    model, metrics, params = train(data, args.seed, args.c_value)
    print_metrics(metrics)

    import joblib

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)
    (model_dir / "metrics.json").write_text(json.dumps({**params, **metrics}, indent=2), encoding="utf-8")

    print()
    print("VERTEX-TRAIN COMPLETE (this also runs fine on your laptop)")
    print(f"  model   : {model_path}")
    print(f"  metrics : {model_dir / 'metrics.json'}")
    print("  In the cloud, AIP_MODEL_DIR points at Cloud Storage, and Vertex")
    print("  picks this file up automatically. Same code, different address.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lab 6 - one training pipeline, two backends (MLflow locally, Vertex AI in the cloud)"
    )
    parser.add_argument("--mode", choices=["local", "vertex", "vertex-train"], default="local",
                        help="local = MLflow on your machine (free). vertex = Google Cloud job (needs credentials).")
    parser.add_argument("--rows", type=int, default=6000, help="synthetic rows to train on")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0, help="inverse regularisation strength")
    parser.add_argument("--project", type=str, default="", help="Google Cloud project id (vertex mode)")
    parser.add_argument("--region", type=str, default=REGION, help="Google Cloud region (vertex mode)")
    parser.add_argument("--bucket", type=str, default="", help="gs:// bucket for staging (vertex mode)")
    parser.add_argument("--tracking-uri", type=str, default="", help="MLflow tracking URI (local mode)")
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("AIP_MODEL_DIR", ""),
                        help="where to write model.joblib (vertex-train mode)")
    args = parser.parse_args()

    print("=" * 70)
    print("LAB 6 - FROM NOTEBOOK SCRIPT TO MANAGED PIPELINE")
    print("=" * 70)
    print(f"  mode : {args.mode}")

    if args.mode == "local":
        return run_local(args)
    if args.mode == "vertex-train":
        if not args.model_dir:
            args.model_dir = str(Path.cwd() / "artifacts")
        return run_vertex_train(args)
    return run_vertex(args)


if __name__ == "__main__":
    sys.exit(main())
