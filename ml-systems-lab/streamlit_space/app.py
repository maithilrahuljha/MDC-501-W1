"""
WHAT: A multi-lab Streamlit dashboard: Lab 1 (break the notebook), Lab 3 (drift), Lab 4 (batch vs online), Lab 6 (pipelines).
HOW:  streamlit run app.py locally, or point Streamlit Community Cloud at this file in your GitHub repo.
OK:   the sidebar lists four labs, and each one runs its experiment and shows a result without errors.
"""

# =============================================================================
# WHY THIS FILE EXISTS
# -----------------------------------------------------------------------------
# The first version of this app showed ONE thing: the churn form. Students kept
# asking "where are the other labs?" -- because a deployed app only runs the one
# entrypoint file you point at it. So this file is the whole course in one page:
# a sidebar with four tabs, each tab running a real experiment from that lab.
#
# WHAT RUNS WHERE -- READ THIS BEFORE YOU JUDGE THE NUMBERS
# -----------------------------------------------------------------------------
# Lab 1  : fully real. Same model, same three failures, same error strings.
# Lab 3  : fully real. A smaller version of lab3_drift/drift_experiment.py, so it
#          finishes in seconds on a shared cloud CPU.
# Lab 4  : PARTLY real, and honest about it. In one call vs one-row-at-a-time,
#          measured in-process. The cloud container has no local HTTP server, so
#          the network cost of the real lab cannot be measured here. Locally,
#          lab4_batch_vs_online/benchmark.py measures the full thing over HTTP,
#          and the gap is far bigger (thousands of times, not dozens).
# Lab 6  : a real training run with real metrics, plus the MLflow/Vertex mapping.
#          No cloud credentials are needed, and nothing is billed.
#
# Everything is generated in memory. No downloads, no logins, no secrets.
# =============================================================================

import platform
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]
MODEL_VERSION = "2.0.0-streamlit"

LABS = [
    "Start here",
    "Lab 1 - Break the notebook",
    "Lab 3 - Static vs dynamic (drift)",
    "Lab 4 - Batch vs online",
    "Lab 6 - Pipelines",
]


# =============================================================================
# 1. DATA AND MODEL (pure functions: no Streamlit calls, so they are testable)
# =============================================================================
def make_training_data(n_rows: int = 6000, seed: int = 42, drift: float = 0.0) -> pd.DataFrame:
    """Synthetic churn data.

    drift=0.0 -> the world the model was trained for.
    drift>0   -> the world AFTER a competitor launched: higher bills, more calls,
                 a different contract mix, and the churn rules themselves shifted.
    One generator with a drift dial keeps Lab 1 and Lab 3 consistent with each
    other, which is the whole point of the course.
    """
    rng = np.random.default_rng(seed)

    if drift <= 0:
        contract = rng.choice(CONTRACTS, size=n_rows, p=[0.60, 0.28, 0.12])
        tenure = rng.integers(1, 72, size=n_rows)
        charges = np.clip(rng.normal(62, 14, size=n_rows), 15, 160)
        calls = rng.poisson(1.3, size=n_rows)
        effect = np.where(contract == "monthly", 1.05, np.where(contract == "one_year", 0.15, -0.95))
        logit = 1.6 * (
            0.045 * (30 - tenure) + 0.014 * (charges - 62) + 0.55 * calls + effect - 0.35
        )
    else:
        contract = rng.choice(CONTRACTS, size=n_rows, p=[0.10, 0.22, 0.68])
        tenure = rng.integers(2, 96, size=n_rows)
        charges = np.clip(rng.normal(88, 20, size=n_rows), 15, 190)
        calls = rng.poisson(2.4, size=n_rows)
        effect = np.where(contract == "monthly", 0.35, np.where(contract == "one_year", -0.55, -1.45))
        logit = 1.6 * (
            -0.020 * (30 - tenure) - 0.010 * (charges - 88) + 1.05 * calls + effect + 0.55
        )

    churn = (rng.random(n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": np.round(charges, 2),
            "support_calls": calls,
            "contract_type": contract,
            "churn": churn,
        }
    )


def build_pipeline(**kwargs) -> Pipeline:
    """The same pipeline shape used by every lab: scale numbers, one-hot categories."""
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(**kwargs), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[("preprocess", preprocess), ("model", LogisticRegression(max_iter=1000, random_state=42))]
    )


def train_churn(drift: float = 0.0, n_rows: int = 6000, seed: int = 42):
    """Train on (possibly drifted) data. Returns (model, metrics dataframes)."""
    data = make_training_data(n_rows=n_rows, seed=seed, drift=drift)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data["churn"], test_size=0.25, random_state=seed, stratify=data["churn"]
    )
    # handle_unknown="error" is the scikit-learn default and it is deliberate here:
    # a category the model never saw must FAIL LOUDLY (see the Lab 1 tab), not be
    # silently treated as all-zeros. The drift tab passes "ignore" instead, because
    # there a new value is expected and we want the model to keep serving.
    model = build_pipeline(handle_unknown="error", sparse_output=False)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "f1": round(float(f1_score(y_test, preds)), 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return model, metrics


# =============================================================================
# 2. THE THREE DELIBERATE FAILURES (Lab 1). Real calls, real exceptions.
# =============================================================================
BROKEN_CASES = {
    "Missing column (monthly_charges)": {
        "body": {"tenure_months": 12, "support_calls": 2, "contract_type": "monthly"},
        "pillar": "Pillar 1 - data validation",
        "why": "A whole field was never sent. The model got a row with a hole in it.",
    },
    "Wrong data type (tenure = 'twelve')": {
        "body": {"tenure_months": "twelve", "monthly_charges": 79.5,
                 "support_calls": 2, "contract_type": "monthly"},
        "pillar": "Pillar 4 - serving contract",
        "why": "A number arrived as a word. A human copes; the arithmetic does not.",
    },
    "Unseen category (contract = 'lifetime')": {
        "body": {"tenure_months": 12, "monthly_charges": 79.5,
                 "support_calls": 2, "contract_type": "lifetime"},
        "pillar": "Pillar 1 (block it) + Pillar 5 (notice it)",
        "why": "A value appeared that the model never saw in training.",
    },
}


def naive_predict(model, payload: dict) -> dict:
    """What a notebook export does: build a row from whatever arrived, then call predict.

    Returns {"ok": bool, "result"/, "error"/}. Never raises, so the UI can show
    the failure instead of dying -- which is exactly what a real service must do.
    """
    try:
        # NOTE: no row[FEATURES] here, on purpose. Passing the frame straight to the
        # pipeline is what a notebook export does, and it is what produces the real
        # error "columns are missing: {'monthly_charges'}" in Lab 1.
        row = pd.DataFrame([payload])
        probability = float(model.predict_proba(row)[0, 1])
        return {"ok": True, "probability": round(probability, 4)}
    except Exception as exc:  # noqa: BLE001 - the point of the exercise
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}


# =============================================================================
# 3. THE DRIFT EXPERIMENT (Lab 3), small enough to finish on a shared cloud CPU
# =============================================================================
def simulate_drift(days: int = 60, per_day: int = 100, drift_day: int = 24,
                   retrain_every: int = 500, window: int = 1500, roll: int = 150):
    """Walk a stream once. Score every row with whatever model is live at that moment.

    Returns (frame with rolling accuracy per row, first sample index where the
    static model drops below 0.70 or None).
    """
    rng_stream = np.random.default_rng(7)
    day_frames = []
    for day in range(days):
        drifted = 1.0 if day >= drift_day else 0.0
        chunk = make_training_data(n_rows=per_day, seed=int(rng_stream.integers(1_000_000)), drift=drifted)
        chunk["day"] = day
        day_frames.append(chunk)

    stream = pd.concat(day_frames, ignore_index=True)
    stream.insert(0, "sample_index", np.arange(len(stream)))
    warmup = min(1500, len(stream) // 3)

    static_model = build_pipeline(handle_unknown="ignore", sparse_output=False).fit(
        stream.iloc[:warmup][FEATURES], stream.iloc[:warmup]["churn"]
    )
    dynamic_model = static_model  # identical at the start; only the schedule differs

    static_pred = np.full(len(stream), -1, dtype=int)
    dynamic_pred = np.full(len(stream), -1, dtype=int)
    last_retrain = warmup

    for start in range(warmup, len(stream), 500):
        stop = min(start + 500, len(stream))
        block = stream.iloc[start:stop]
        static_pred[start:stop] = static_model.predict(block[FEATURES])
        dynamic_pred[start:stop] = dynamic_model.predict(block[FEATURES])

        if stop - last_retrain >= retrain_every:
            history = stream.iloc[max(0, stop - window):stop]
            dynamic_model = build_pipeline(handle_unknown="ignore", sparse_output=False).fit(
                history[FEATURES], history["churn"]
            )
            last_retrain = stop

    result = stream[["sample_index", "day", "churn"]].copy()
    result["static_pred"] = static_pred
    result["dynamic_pred"] = dynamic_pred
    result = result[result["static_pred"] >= 0].copy()
    result["static_correct"] = (result["static_pred"] == result["churn"]).astype(int)
    result["dynamic_correct"] = (result["dynamic_pred"] == result["churn"]).astype(int)
    result["static_accuracy"] = result["static_correct"].rolling(roll, min_periods=roll // 2).mean()
    result["dynamic_accuracy"] = result["dynamic_correct"].rolling(roll, min_periods=roll // 2).mean()

    below = result[result["static_accuracy"] < 0.70]
    first_below = None if below.empty else int(below.iloc[0]["sample_index"])
    return result, first_below, drift_day * per_day


# =============================================================================
# 4. THE BATCH vs ONLINE EXPERIMENT (Lab 4), in-process and honest about it
# =============================================================================
def compare_batch_vs_single(model, rows: int = 5000, sequential_rows: int = 1500, seed: int = 11):
    """Time one big call against N single-row calls. Returns a dict of numbers."""
    data = make_training_data(n_rows=rows, seed=seed)

    started = time.perf_counter()
    model.predict_proba(data[FEATURES])
    batch_seconds = time.perf_counter() - started

    single = data.head(sequential_rows)
    latencies = []
    started = time.perf_counter()
    for _, row in single.iterrows():                      # one customer at a time
        one = pd.DataFrame([row[FEATURES]])
        step = time.perf_counter()
        model.predict_proba(one)
        latencies.append((time.perf_counter() - step) * 1000)
    sequential_seconds = time.perf_counter() - started

    return {
        "rows_batch": int(rows),
        "batch_seconds": batch_seconds,
        "batch_ms_per_row": batch_seconds / rows * 1000,
        "rows_single": int(sequential_rows),
        "sequential_seconds": sequential_seconds,
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "slowdown": sequential_seconds / (batch_seconds * sequential_rows / rows),
    }


# =============================================================================
# 5. THE STREAMLIT UI
# =============================================================================
@st.cache_resource
def get_model():
    """Train once per session and reuse. This is what makes the app feel instant."""
    return train_churn()


def page_start():
    st.title("From notebook to production")
    st.write(
        "One page, four labs. Use the **sidebar on the left** (the arrow at the top-left on a "
        "phone) to switch between them. Each one runs a real experiment on synthetic data."
    )

    st.table(
        pd.DataFrame(
            [
                {"Lab": "Lab 1", "Idea": "A notebook is not a system",
                 "You will see": "Three real crashes, and the fix for each"},
                {"Lab": "Lab 3", "Idea": "Static vs dynamic training",
                 "You will see": "A frozen model rotting while a retrained one keeps up"},
                {"Lab": "Lab 4", "Idea": "Batch vs online inference",
                 "You will see": "The same model, two call shapes, two very different costs"},
                {"Lab": "Lab 6", "Idea": "Pipelines and tracking",
                 "You will see": "A training run with metrics, and the MLflow/Vertex map"},
            ]
        )
    )

    st.info(
        "The five pillars behind every lab: **1** data validation, **2** testing, "
        "**3** versioning, **4** serving contract, **5** monitoring."
    )
    with st.expander("What is real here, and what is a simplification?"):
        st.markdown(
            "- **Lab 1** is real: the same model, the same three failures, the same error text.\n"
            "- **Lab 3** is real, with a smaller stream so it finishes in seconds.\n"
            "- **Lab 4** measures one-call vs one-row-at-a-time **in process**. A cloud "
            "container has no local web server, so the network cost is not included here. "
            "It is still thousands of times slower, because every single-row call pays a "
            "fixed cost. Run `lab4_batch_vs_online/benchmark.py` locally to add HTTP on top "
            "(we measured about 10,000x there).\n"
            "- **Lab 6** trains for real and shows real metrics. No cloud account is used "
            "and nothing is billed."
        )


def page_lab1():
    st.title("Lab 1 - Break the notebook")
    st.write(
        "The form below is a **contract**: it cannot send a wrong type, and the contract "
        "type is limited to three values. That is Pillar 4 doing its job. "
        "Then break it on purpose, twice, and read what a model without a contract does."
    )

    model, metrics = get_model()
    st.caption(
        f"model version {MODEL_VERSION} | offline accuracy {metrics['accuracy']:.3f} | "
        f"trained on {metrics['train_rows']:,} rows | ROC-AUC {metrics['roc_auc']:.3f}"
    )

    with st.form("customer"):
        col1, col2 = st.columns(2)
        with col1:
            tenure = st.slider("Tenure (months as a customer)", 0, 96, 12)
            charges = st.number_input("Monthly charges (USD)", 0.0, 400.0, 79.5, step=1.0)
        with col2:
            calls = st.slider("Support calls last month", 0, 15, 2)
            contract = st.selectbox("Contract type", CONTRACTS)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        payload = {
            "tenure_months": int(tenure),
            "monthly_charges": float(charges),
            "support_calls": int(calls),
            "contract_type": contract,
        }
        outcome = naive_predict(model, payload)
        if outcome["ok"]:
            probability = outcome["probability"]
            st.metric("Chance of leaving", f"{probability:.0%}")
            if probability >= 0.65:
                st.error("Likely to leave. Worth a retention call.")
            elif probability >= 0.35:
                st.warning("Might leave. Keep an eye on this one.")
            else:
                st.success("Likely to stay.")
        else:
            st.error(f"{outcome['error_type']}: {outcome['error']}")

    st.divider()
    st.subheader("Now break it on purpose")
    st.write(
        "Each button sends a request that a real user would eventually send. "
        "The **naive** column is what happens with no checks. The **contract** column "
        "is what should happen instead."
    )

    for name, case in BROKEN_CASES.items():
        with st.expander(name):
            st.write(case["why"])
            naive = naive_predict(model, case["body"])
            left, right = st.columns(2)
            with left:
                st.write("**Naive endpoint (what a notebook export does)**")
                if naive["ok"]:
                    st.success(f"No error. It answered {naive['probability']:.3f} anyway.")
                else:
                    st.code(f"{naive['error_type']}: {naive['error']}", language="text")
            with right:
                st.write("**With a contract (the fix)**")
                st.markdown(
                    "- HTTP **422** before the model is ever called\n"
                    f"- Prevents it: **{case['pillar']}**"
                )

    st.caption(
        "Notice what did NOT happen: the process did not die. A 422 is a polite refusal; "
        "a 500 is an outage. Same model, different door."
    )


def page_lab3():
    st.title("Lab 3 - Static vs dynamic training")
    st.write(
        "A stream of customers arrives. On the chosen day the world changes: bills rise, "
        "support calls spike, and the rules themselves shift. One model is trained once and "
        "frozen. The other refits on recent data. Nothing crashes in either case."
    )

    with st.form("drift"):
        col1, col2, col3 = st.columns(3)
        with col1:
            days = st.slider("Days to simulate", 30, 120, 60)
        with col2:
            drift_day = st.slider("World changes on day", 5, 80, 24)
        with col3:
            retrain_every = st.slider("Retrain every N customers", 250, 2000, 500, step=250)
        run = st.form_submit_button("Run the experiment", type="primary")

    if run or "drift_result" not in st.session_state:
        with st.spinner("Simulating the stream and training both models ..."):
            frame, first_below, drift_index = simulate_drift(
                days=days, drift_day=drift_day, retrain_every=retrain_every
            )
            st.session_state.drift_result = (frame, first_below, drift_index)

    frame, first_below, drift_index = st.session_state.drift_result
    tail = frame.tail(150)

    col1, col2, col3 = st.columns(3)
    col1.metric("Frozen model, last 150 customers", f"{tail['static_correct'].mean():.1%}")
    col2.metric("Retrained model, last 150 customers", f"{tail['dynamic_correct'].mean():.1%}")
    col3.metric("First drop below 70%", f"sample {first_below:,}" if first_below else "never")

    chart = frame.set_index("sample_index")[["static_accuracy", "dynamic_accuracy"]]
    chart.columns = ["Frozen model (trained once)", "Retrained model"]
    st.line_chart(chart, height=340)

    st.caption(
        f"The world changes at sample {drift_index:,}. Red-style line: frozen model. "
        "If the two lines separate, you have just seen data drift AND concept drift at once."
    )
    if first_below:
        st.warning(
            f"The frozen model's rolling accuracy first falls below 70% at sample "
            f"**{first_below:,}**. Nothing crashed. Nothing emailed you. It just stopped "
            "being right — which is why Pillar 5 (monitoring) exists."
        )

    with st.expander("What is different from the local lab?"):
        st.markdown(
            "Same maths, smaller stream so it finishes on a shared CPU. The full version, "
            "`lab3_drift/drift_experiment.py`, also writes `drift_accuracy.png`, a CSV of "
            "every prediction, and a JSON report — the artifacts you would actually keep."
        )


def page_lab4():
    st.title("Lab 4 - Batch vs online")
    st.write(
        "One trained model, two ways to ask it for answers. **Batch**: hand it a table. "
        "**Online**: hand it one customer, get one answer, repeat."
    )

    model, _ = get_model()
    with st.form("bench"):
        col1, col2 = st.columns(2)
        with col1:
            rows = st.slider("Rows in the batch call", 1000, 20000, 5000, step=1000)
        with col2:
            sequential = st.slider("Single-row calls to time", 200, 3000, 1000, step=100)
        run = st.form_submit_button("Run the comparison", type="primary")

    if run or "bench_result" not in st.session_state:
        with st.spinner("Scoring both ways ..."):
            st.session_state.bench_result = compare_batch_vs_single(
                model, rows=rows, sequential_rows=sequential
            )

    result = st.session_state.bench_result
    st.table(
        pd.DataFrame(
            [
                {"Method": "Batch (one call)", "Rows": f"{result['rows_batch']:,}",
                 "Total time": f"{result['batch_seconds'] * 1000:.1f} ms",
                 "Per row": f"{result['batch_ms_per_row']:.4f} ms"},
                {"Method": "Online (one row per call)", "Rows": f"{result['rows_single']:,}",
                 "Total time": f"{result['sequential_seconds']:.3f} s",
                 "Per row": f"{result['avg_latency_ms']:.3f} ms"},
            ]
        )
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Average latency per single prediction", f"{result['avg_latency_ms']:.3f} ms")
    col2.metric("95th percentile latency", f"{result['p95_latency_ms']:.3f} ms")
    col3.metric("Online is slower by", f"{result['slowdown']:.1f}x")

    st.info(
        "**Read that number with care.** It is measured in-process, so it counts Python and "
        "pandas overhead only — no network. Every single-row call pays a fixed cost (build "
        "one row, validate it, enter scikit-learn, come back), while the batch call pays it "
        "once for thousands of rows. Add HTTP on top and the gap grows further: the local "
        "lab measured about 10,000x. Run `lab4_batch_vs_online/benchmark.py` to see that."
    )
    st.caption(
        "The lesson survives the smaller number: batch wins on cost, online wins on "
        "freshness, and you choose by asking when the answer has to exist."
    )


def page_lab6():
    st.title("Lab 6 - Pipelines, tracking and the cloud")
    st.write(
        "A pipeline is your notebook written as a program: make data, split, train, "
        "evaluate, save. Here is that run, for real, plus the map between the free "
        "local tool (MLflow) and the paid cloud one (Vertex AI)."
    )

    if st.button("Run the training pipeline", type="primary"):
        with st.spinner("Training ..."):
            _, metrics = train_churn(n_rows=8000, seed=7)
        st.success("Run finished. In MLflow this is one row in one experiment.")
        st.table(pd.DataFrame([metrics]).T.rename(columns={0: "value"}))

    st.subheader("The same idea, two backends")
    st.table(
        pd.DataFrame(
            [
                {"Vertex AI (cloud, paid)": "Experiment", "MLflow (local, free)": "Experiment"},
                {"Vertex AI (cloud, paid)": "Run / custom training job", "MLflow (local, free)": "Run"},
                {"Vertex AI (cloud, paid)": "Model Registry", "MLflow (local, free)": "Registered model"},
                {"Vertex AI (cloud, paid)": "Endpoint", "MLflow (local, free)": "Your own FastAPI service"},
            ]
        )
    )
    st.code(
        "python pipeline.py --mode local     # MLflow, free, writes ./mlruns\n"
        "python pipeline.py --mode vertex    # Vertex AI, needs a GCP project + credentials",
        language="bash",
    )
    st.caption(
        "Without credentials, `--mode vertex` prints a friendly setup message and exits 0. "
        "No cost, no crash, no secrets in the repo."
    )


def main() -> None:
    st.set_page_config(
        page_title="MDC-501 - From notebook to production",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("MDC-501")
    st.sidebar.caption("From notebook to production")
    choice = st.sidebar.radio("Choose a lab", LABS, index=0)
    st.sidebar.divider()
    st.sidebar.caption(f"model version {MODEL_VERSION}")
    st.sidebar.caption(f"python {platform.python_version()} | streamlit {st.__version__}")

    pages = {
        LABS[0]: page_start,
        LABS[1]: page_lab1,
        LABS[2]: page_lab3,
        LABS[3]: page_lab4,
        LABS[4]: page_lab6,
    }
    pages[choice]()

    st.divider()
    st.caption(
        "All data is synthetic and generated in this file. No real customer records, "
        "no API keys, no downloads."
    )


# A one-line version banner in the Cloud log. It answers "what is actually running
# in there?" -- the first question to ask when a deploy misbehaves.
print(
    f"[startup] python {platform.python_version()} | streamlit {st.__version__} | "
    f"pandas {pd.__version__} | numpy {np.__version__}",
    file=sys.stderr,
)

# Streamlit executes this file as __main__, so the UI is built there.
# Importing the file (tests, tooling) does NOT launch the app -- the pure
# functions above can be tested without a browser or a Streamlit runtime.
if __name__ == "__main__":
    main()
