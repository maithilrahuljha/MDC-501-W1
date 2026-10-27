"""
WHAT: A tiny Streamlit web app that serves the churn model -- the zero-cost backup to Hugging Face Spaces (Part H, Plan B).
HOW:  streamlit run app.py locally, or point Streamlit Community Cloud at this file in your GitHub repo.
OK:   a page loads with a form; pressing Submit shows a churn percentage and a version line.
"""

# =============================================================================
# WHY THIS FILE EXISTS
# -----------------------------------------------------------------------------
# Hugging Face Spaces used to be free for everyone. In 2026 Hugging Face changed
# the rules: Gradio and Docker Spaces on the free CPU tier now usually require a
# paid PRO plan. Free accounts in good standing can still run up to two Gradio
# Spaces on "ZeroGPU", but brand-new accounts often only see "Static" as free.
#
# So this lab gives you two roads to the same destination -- a public URL:
#   Plan A : Hugging Face Spaces   (hf_space/app.py)      -- see Part H
#   Plan B : Streamlit Community Cloud (this file)        -- see Part H, Plan B
#
# Streamlit Community Cloud is free, needs only a GitHub account, and deploys
# straight from the repo you created in Part G. That is why it is the backup.
#
# The model logic is identical to hf_space/app.py on purpose. Only the buttons
# and boxes change. That is the lesson: the model is portable; the packaging is
# what you rewrite.
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]
MODEL_VERSION = "1.0.0-streamlit"


def make_training_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic churn data, generated in memory. No downloads, no secrets."""
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


@st.cache_resource  # train ONCE, reuse for every visitor (this is why the app feels instant)
def load_model() -> Pipeline:
    data = make_training_data()
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(data[FEATURES], data["churn"])
    return model


MODEL = load_model()
PREDICTIONS = {"count": 0}

st.set_page_config(page_title="Churn predictor - MDC-501", page_icon="📉")
st.title("Will this customer leave?")
st.write(
    "A real model, served in a browser. Four inputs, one probability. "
    "This is the same model as Lab 1 -- but with a **contract** around it."
)

with st.form("customer"):
    tenure = st.number_input("Tenure (months as a customer)", min_value=0, max_value=120, value=12, step=1)
    charges = st.number_input("Monthly charges (USD)", min_value=0.0, max_value=1000.0, value=79.5, step=1.0)
    calls = st.number_input("Support calls last month", min_value=0, max_value=100, value=2, step=1)
    contract = st.selectbox("Contract type", CONTRACTS)
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame(
        [{
            "tenure_months": int(tenure),
            "monthly_charges": float(charges),
            "support_calls": int(calls),
            "contract_type": contract,
        }]
    )
    probability = float(MODEL.predict_proba(row[FEATURES])[0, 1])
    PREDICTIONS["count"] += 1

    st.metric("Chance of leaving", f"{probability:.0%}")
    if probability >= 0.65:
        st.error("Likely to leave. This customer deserves a phone call.")
    elif probability >= 0.35:
        st.warning("Might leave. Keep an eye on them.")
    else:
        st.success("Likely to stay.")

    st.write(
        f"Tenure {int(tenure)} months, bill ${float(charges):.2f}, "
        f"{int(calls)} support calls, {contract.replace('_', ' ')} contract."
    )

st.divider()
st.caption(
    f"Model version {MODEL_VERSION} | logistic regression | trained on synthetic data | "
    f"always on | predictions so far this session: {PREDICTIONS['count']}"
)
st.caption(
    "Built for MDC-501: From Notebook to Production. "
    "The four boxes cannot send a wrong type -- that is the contract doing its job."
)
