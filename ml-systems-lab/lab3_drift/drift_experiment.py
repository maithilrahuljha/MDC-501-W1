"""
WHAT: Simulates two years of customer traffic, drifts the world halfway through, and compares a static model with a retrained one.
HOW:  python drift_experiment.py            (local)   |   open in Colab and press Run All (defaults are fine)
OK:   prints "EXPERIMENT COMPLETE", a sample index where static accuracy drops under 70%, and writes outputs/drift_accuracy.png.
"""

# =============================================================================
# THE IDEA IN ONE PARAGRAPH
# -----------------------------------------------------------------------------
# A STATIC model is trained once and frozen. A DYNAMIC model is retrained on a
# schedule with fresh data. In a stable world the two behave the same. In a
# changing world -- which is every real world -- the static model quietly rots.
# "Data drift" is when the inputs change. "Concept drift" is when the
# relationship between inputs and the answer changes. This file injects both
# and shows you exactly what each one costs.
#
# HOW TO READ THE OUTPUT
# -----------------------------------------------------------------------------
#   sample number  = the x-axis. Imagine 100 customers arriving per day.
#   accuracy       = of the last 200 customers, how many did we label correctly.
#   drift starts at day 40 (sample 4000). Watch the blue line fall off a cliff.
# =============================================================================

import argparse
import json
import sys
from pathlib import Path

# --- matplotlib: work both in a notebook (Colab) and in a plain terminal ------
import matplotlib
import numpy as np
import pandas as pd

IN_NOTEBOOK = "ipython" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # no screen needed; we save a PNG instead
import matplotlib.pyplot as plt  # noqa: E402

try:  # only exists inside Jupyter/Colab
    from IPython.display import Image, display  # type: ignore

    HAVE_IPYTHON_DISPLAY = True
except ImportError:
    HAVE_IPYTHON_DISPLAY = False


from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]

DEFAULTS = {
    "days": 100,
    "per_day": 100,
    "drift_day": 40,
    "warmup": 2000,
    "retrain_every": 500,
    "window": 2000,
    "roll": 200,
    "seed": 7,
}


# =============================================================================
# 1. THE WORLD (this is your "data source". No downloads, no login.)
# =============================================================================
def generate_world(day: int, per_day: int, drift_day: int, rng: np.random.Generator):
    """Return one day of customers. Before drift_day it is World A; after, World B."""
    drifted = day >= drift_day

    if not drifted:
        # --- WORLD A: the world the model was trained for -------------------
        contract = rng.choice(CONTRACTS, size=per_day, p=[0.60, 0.28, 0.12])
        tenure = rng.integers(2, 72, size=per_day)
        charges = np.round(np.clip(rng.normal(62, 14, size=per_day), 15, 160), 2)
        calls = rng.poisson(1.3, size=per_day)

        contract_effect = np.where(contract == "monthly", 1.05,
                                   np.where(contract == "one_year", 0.15, -0.95))
        logit = 1.6 * (
            0.045 * (30 - tenure)     # newer customers leave more
            + 0.014 * (charges - 62)  # paying more hurts
            + 0.55 * calls            # being annoyed hurts
            + contract_effect
            - 0.35
        )
    else:
        # --- WORLD B: a competitor launched, prices rose, people changed ---
        # 1) COVARIATE SHIFT: different customers, different mix, higher bills.
        contract = rng.choice(CONTRACTS, size=per_day, p=[0.10, 0.22, 0.68])
        tenure = rng.integers(2, 96, size=per_day)
        charges = np.round(np.clip(rng.normal(88, 20, size=per_day), 15, 190), 2)
        calls = rng.poisson(2.4, size=per_day)

        # 2) CONCEPT SHIFT: the relationship itself flipped. Price is no longer
        #    the villain (everyone got expensive); support calls now dominate,
        #    and long tenure stopped being protective because of a price hike.
        contract_effect = np.where(contract == "monthly", 0.35,
                                   np.where(contract == "one_year", -0.55, -1.45))
        logit = 1.6 * (
            -0.020 * (30 - tenure)     # <-- sign flipped vs World A
            - 0.010 * (charges - 88)   # <-- sign flipped vs World A
            + 1.05 * calls             # <-- much stronger weight
            + contract_effect
            + 0.55
        )

    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = (rng.random(per_day) < probability).astype(int)

    return pd.DataFrame(
        {
            "day": day,
            "tenure_months": tenure,
            "monthly_charges": charges,
            "support_calls": calls,
            "contract_type": contract,
            "churn": churn,
        }
    )


def build_stream(days: int, per_day: int, drift_day: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [generate_world(day, per_day, drift_day, rng) for day in range(days)]
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "sample_index", np.arange(len(df)))
    return df


# =============================================================================
# 2. THE MODEL (identical for both strategies -- only the TRAINING SCHEDULE differs)
# =============================================================================
def new_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )


def fit(df: pd.DataFrame) -> Pipeline:
    model = new_model()
    model.fit(df[FEATURES], df["churn"])
    return model


# =============================================================================
# 3. THE EXPERIMENT
# =============================================================================
def run_experiment(df: pd.DataFrame, warmup: int, retrain_every: int, window: int, roll: int):
    """Walk the stream once. Score every row with whatever model is 'live'."""
    n = len(df)
    warmup = min(warmup, n // 3)

    # Both models start identical: trained on the first `warmup` rows.
    static_model = fit(df.iloc[:warmup])
    dynamic_model = fit(df.iloc[:warmup])

    static_pred = np.full(n, -1, dtype=int)
    dynamic_pred = np.full(n, -1, dtype=int)
    retrain_points = [warmup]

    batch_size = 500  # score in chunks, the way a real batch job would
    last_retrain = warmup
    for start in range(warmup, n, batch_size):
        stop = min(start + batch_size, n)
        chunk = df.iloc[start:stop]

        # Every row gets scored by the model that exists RIGHT NOW.
        static_pred[start:stop] = static_model.predict(chunk[FEATURES])
        dynamic_pred[start:stop] = dynamic_model.predict(chunk[FEATURES])

        # Labels for this chunk "arrive" after the fact (here: immediately).
        # The dynamic model uses them to refresh itself on the recent window.
        if stop - last_retrain >= retrain_every:
            history = df.iloc[max(0, stop - window):stop]
            dynamic_model = fit(history)      # <-- the one line that makes it "dynamic"
            last_retrain = stop
            retrain_points.append(stop)

    results = df[["sample_index", "day", "churn"]].copy()
    results["static_pred"] = static_pred
    results["dynamic_pred"] = dynamic_pred

    scored = results[results["static_pred"] >= 0].copy()
    scored["static_correct"] = (scored["static_pred"] == scored["churn"]).astype(int)
    scored["dynamic_correct"] = (scored["dynamic_pred"] == scored["churn"]).astype(int)

    scored["static_accuracy"] = scored["static_correct"].rolling(roll, min_periods=roll // 2).mean()
    scored["dynamic_accuracy"] = scored["dynamic_correct"].rolling(roll, min_periods=roll // 2).mean()

    return scored, retrain_points, warmup


def first_index_below(scored: pd.DataFrame, column: str, threshold: float = 0.70):
    hits = scored[scored[column] < threshold]
    if hits.empty:
        return None
    row = hits.iloc[0]
    return int(row["sample_index"]), int(row["day"]), float(row[column])


def accuracy_before_after(scored: pd.DataFrame, warmup: int, drift_index: int) -> pd.DataFrame:
    rows = []
    for name, start, stop in [
        ("before drift", warmup, drift_index),
        ("after drift ", drift_index, len(scored)),
    ]:
        part = scored[(scored["sample_index"] >= start) & (scored["sample_index"] < stop)]
        if part.empty:
            continue
        rows.append(
            {
                "period": name,
                "rows": len(part),
                "static_accuracy": round(float(part["static_correct"].mean()), 3),
                "dynamic_accuracy": round(float(part["dynamic_correct"].mean()), 3),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 4. REPORTING
# =============================================================================
def make_plot(scored: pd.DataFrame, drift_index: int, out_path: Path, title_note: str) -> None:
    plt.figure(figsize=(11, 5.5))
    plt.plot(scored["sample_index"], scored["static_accuracy"], color="#c0392b",
             linewidth=2, label="Static model (trained once, never updated)")
    plt.plot(scored["sample_index"], scored["dynamic_accuracy"], color="#1f77b4",
             linewidth=2, label="Dynamic model (retrained every window)")
    plt.axhline(0.70, color="black", linestyle=":", linewidth=1.2)
    plt.text(scored["sample_index"].iloc[3], 0.712, "70% alert line", fontsize=9, color="black")
    plt.axvline(drift_index, color="gray", linestyle="--", linewidth=1.2)
    plt.text(drift_index + 60, 0.93, "world changes here", fontsize=9, color="gray")
    plt.ylim(0.45, 1.0)
    plt.xlabel("Sample number (100 customers per day)")
    plt.ylabel("Rolling accuracy (last 200 customers)")
    plt.title("Lab 3: the same model, two training strategies\n" + title_note)
    plt.legend(loc="lower left", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140)
    print(f"\nPlot saved to: {out_path}")

    if IN_NOTEBOOK and HAVE_IPYTHON_DISPLAY:
        display(Image(filename=str(out_path)))  # shows inline in Colab
    else:
        plt.show()  # harmless no-op with the Agg backend
    plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 3 - static vs dynamic training under data drift")
    parser.add_argument("--days", type=int, default=DEFAULTS["days"], help="how many simulated days")
    parser.add_argument("--per-day", type=int, default=DEFAULTS["per_day"], help="customers per day")
    parser.add_argument("--drift-day", type=int, default=DEFAULTS["drift_day"], help="the world changes on this day")
    parser.add_argument("--warmup", type=int, default=DEFAULTS["warmup"], help="rows used for the first training")
    parser.add_argument("--retrain-every", type=int, default=DEFAULTS["retrain_every"], help="dynamic model retrain cadence")
    parser.add_argument("--window", type=int, default=DEFAULTS["window"], help="how many recent rows the retrain uses")
    parser.add_argument("--roll", type=int, default=DEFAULTS["roll"], help="rolling accuracy window")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="random seed")
    parser.add_argument("--outdir", type=str, default="", help="where to write outputs (default: ./outputs)")
    args = parser.parse_args()

    try:
        here = Path(__file__).resolve().parent
    except NameError:  # some notebooks do not define __file__
        here = Path.cwd()
    outdir = Path(args.outdir) / "outputs" if args.outdir else here / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("LAB 3 - STATIC vs DYNAMIC TRAINING UNDER DRIFT")
    print("=" * 74)
    print(f"  simulated days        : {args.days}")
    print(f"  customers per day     : {args.per_day}")
    print(f"  world changes on day  : {args.drift_day}")
    print(f"  retrain cadence       : every {args.retrain_every} customers")
    print(f"  retrain window        : last {args.window} customers")
    print("  both models are logistic regression. only the schedule differs.")

    df = build_stream(args.days, args.per_day, args.drift_day, args.seed)
    drift_index = args.drift_day * args.per_day
    print(f"\n  total rows streamed   : {len(df):,}")
    print(f"  drift starts at sample: {drift_index:,}")

    scored, retrain_points, warmup = run_experiment(df, args.warmup, args.retrain_every, args.window, args.roll)

    print(f"  first training rows   : {warmup:,}")
    print(f"  dynamic retrains at   : {', '.join(str(p) for p in retrain_points)}")

    summary = accuracy_before_after(scored, warmup, drift_index)
    print("\n" + "-" * 74)
    print("AVERAGE ACCURACY, BEFORE vs AFTER THE WORLD CHANGES")
    print("-" * 74)
    print(summary.to_string(index=False))

    print("\n" + "-" * 74)
    print("THE MOMENT OF TRUTH")
    print("-" * 74)
    hit = first_index_below(scored, "static_accuracy", 0.70)
    if hit is None:
        print("  The static model never fell below 70% accuracy.")
        print("  That means this simulated world did not drift hard enough.")
        print("  Try: python drift_experiment.py --drift-day 30")
    else:
        index, day, acc = hit
        print(f"  Static model accuracy first drops below 70% at SAMPLE INDEX {index:,}.")
        print(f"  That is day {day} of the simulation, at an accuracy of {acc:.1%}.")
        print("  Nothing crashed. No alarm rang. The predictions just stopped being right.")
        print("  THIS is why a model in a notebook is not a production system.")

    dyn_hit = first_index_below(scored, "dynamic_accuracy", 0.70)
    if dyn_hit is None:
        print("\n  The dynamic model never fell below 70%.")
    else:
        print(f"\n  The dynamic model also dipped below 70% at sample {dyn_hit[0]:,} "
              f"(day {dyn_hit[1]}) -- it takes one or two retrains to catch up.")
        rec = scored[(scored["sample_index"] > dyn_hit[0]) & (scored["dynamic_accuracy"] >= 0.75)]
        if not rec.empty:
            print(f"  It was back above 75% by sample {int(rec.iloc[0]['sample_index']):,}. "
                  "That lag is the price of retraining, and it is worth paying.")

    final_window = scored.tail(args.roll)
    print("\n" + "-" * 74)
    print(f"LAST {args.roll} CUSTOMERS")
    print("-" * 74)
    print(f"  static model  : {final_window['static_correct'].mean():.1%}")
    print(f"  dynamic model : {final_window['dynamic_correct'].mean():.1%}")

    scored.to_csv(outdir / "drift_results.csv", index=False)
    summary.to_csv(outdir / "drift_summary.csv", index=False)

    report = {
        "config": vars(args),
        "drift_start_sample_index": drift_index,
        "static_first_below_70pct_sample_index": None if hit is None else hit[0],
        "static_first_below_70pct_day": None if hit is None else hit[1],
        "dynamic_first_below_70pct_sample_index": None if dyn_hit is None else dyn_hit[0],
        "final_static_accuracy": round(float(final_window["static_correct"].mean()), 4),
        "final_dynamic_accuracy": round(float(final_window["dynamic_correct"].mean()), 4),
    }
    (outdir / "drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    make_plot(
        scored,
        drift_index,
        outdir / "drift_accuracy.png",
        "accuracy vs sample number  |  static = red, dynamic = blue",
    )

    print("\n" + "=" * 74)
    print("EXPERIMENT COMPLETE")
    print("=" * 74)
    print(f"  files written to: {outdir}")
    print("    drift_accuracy.png  <- the picture you show your manager")
    print("    drift_results.csv   <- every prediction, for your own analysis")
    print("    drift_summary.csv   <- before/after table")
    print("    drift_report.json    <- machine-readable results")
    print()
    print("  Now say it out loud: the static model did not get worse because")
    print("  the code changed. It got worse because the WORLD changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
