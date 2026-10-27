"""
WHAT: Scores a whole CSV of customers in one vectorised call and writes a predictions file -- the BATCH half of Lab 4.
HOW:  python batch.py            (add --rows 50000 for a bigger file, --input my.csv to use your own)
OK:   prints "BATCH COMPLETE" plus rows scored, total seconds and milliseconds per row.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from online import FEATURES, make_training_data

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_INPUT = DATA_DIR / "serving_batch.csv"


def generate_input_rows(n_rows: int, seed: int = 2024) -> pd.DataFrame:
    """Build n_rows of 'customers to score'. No labels -- this is production input."""
    data = make_training_data(n_rows=n_rows, seed=seed)
    return data[FEATURES].copy()


def ensure_input_file(n_rows: int, path: Path = DEFAULT_INPUT) -> Path:
    """Create the input CSV once, then reuse it so batch and online see the same rows."""
    label = path.name if BASE_DIR not in path.parents else str(path.relative_to(BASE_DIR))
    if path.exists() and len(pd.read_csv(path)) == n_rows:
        print(f"[batch] Reusing existing input file: {label}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_input_rows(n_rows)
    rows.to_csv(path, index=False)
    print(f"[batch] Wrote {n_rows:,} rows to {label}")
    return path


def score_batch(model, rows: pd.DataFrame):
    """THE WHOLE POINT: one call, many rows. The model does the work in one loop."""
    started = time.perf_counter()
    probabilities = model.predict_proba(rows[FEATURES])[:, 1]
    elapsed = time.perf_counter() - started

    predictions = rows.copy()
    predictions["churn_probability"] = np.round(probabilities, 4)
    predictions["churn_label"] = (probabilities >= 0.5).astype(int)
    return predictions, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 4 - batch scoring of a CSV")
    parser.add_argument("--rows", type=int, default=10_000, help="how many rows to score (default 10000)")
    parser.add_argument("--input", type=str, default="", help="optional path to your own CSV")
    parser.add_argument("--seed", type=int, default=2024, help="random seed for synthetic input")
    args = parser.parse_args()

    # Import here so the banner above still prints if the model is missing.
    from online import load_or_train_model

    model, metadata = load_or_train_model()

    if args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.exists():
            print(f"[batch] I cannot find {input_path}")
            print("        Fix: check the path, or run without --input to use the synthetic file.")
            return 1
        print(f"[batch] Using your file: {input_path}")
    else:
        input_path = ensure_input_file(args.rows)

    print(f"[batch] Reading {input_path.name} ...")
    rows = pd.read_csv(input_path)
    n_rows = len(rows)

    print(f"[batch] Model in use: {metadata.get('source', 'unknown')}")
    print(f"[batch] Scoring {n_rows:,} rows in ONE call (no loop, no network) ...")

    predictions, elapsed = score_batch(model, rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUTPUT_DIR / f"predictions_{stamp}.csv"
    predictions.to_csv(out_path, index=False)
    out_label = out_path.name

    summary = {
        "scored_at_utc": stamp,
        "model_version": metadata.get("model_version", "unknown"),
        "rows_scored": int(n_rows),
        "total_seconds": round(elapsed, 4),
        "milliseconds_per_row": round(elapsed / max(n_rows, 1) * 1000, 4),
        "rows_per_second": round(n_rows / elapsed, 1) if elapsed > 0 else None,
        "predicted_churn_rate": round(float(predictions["churn_label"].mean()), 4),
        "output_file": out_label,
    }
    (OUTPUT_DIR / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("First 3 rows of the answer:")
    print(predictions.head(3).to_string(index=False))
    print()
    print("=" * 68)
    print("BATCH COMPLETE")
    print(f"  rows scored        : {n_rows:,}")
    print(f"  total time         : {elapsed:.3f} s")
    print(f"  time per row       : {elapsed / n_rows * 1000:.4f} ms")
    print(f"  rows per second    : {n_rows / elapsed:,.0f}")
    print(f"  predicted churn    : {predictions['churn_label'].mean():.1%} of customers")
    print(f"  predictions file   : outputs/{out_label}")
    print("=" * 68)
    print()
    print("Notice: there is no 'per-request overhead' here. One process, one call,")
    print("10,000 answers. Now run 'python benchmark.py' to see what the same")
    print("10,000 answers cost when they arrive one at a time over HTTP.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
