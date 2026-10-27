"""
WHAT: A tiny Gradio web app that serves the churn model, built to run free on Hugging Face Spaces (Lab 5 in instructions.txt).
HOW:  python app.py   locally, or upload app.py + requirements.txt to a new Hugging Face Space and wait for it to build.
OK:   a page loads with four input controls; moving a slider and clicking Submit shows a churn probability.
"""

# =============================================================================
# IMPORTANT: this app is SELF-CONTAINED.
# It trains its own 3-second model on startup, so there is no model file to
# upload and nothing to configure. Fewer moving parts = fewer ways to fail on
# your first deployment. (Lab 1 is where we show you what breaks. This is
# where we show you what shipping looks like.)
#
# Everything a deployed model needs is here:
#   1. a contract (the four inputs, with limits)
#   2. a version (MODEL_VERSION)
#   3. a health signal (the "Model status" line at the bottom)
#   4. monitoring hooks (every prediction is printed to the Space logs)
# =============================================================================

import os

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["tenure_months", "monthly_charges", "support_calls", "contract_type"]
NUMERIC = ["tenure_months", "monthly_charges", "support_calls"]
CATEGORICAL = ["contract_type"]
CONTRACTS = ["monthly", "one_year", "two_year"]
MODEL_VERSION = "1.0.0-space"

PREDICTION_COUNT = {"n": 0}


def make_training_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Synthetic churn data, generated at startup. No downloads, no secrets."""
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


def train_model() -> Pipeline:
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


print("[startup] Training the model (about 2 seconds) ...")
MODEL = train_model()
print(f"[startup] Ready. Version {MODEL_VERSION}")


def predict(tenure_months, monthly_charges, support_calls, contract_type):
    """One row in, one honest answer out. Also logs the request (Pillar 5)."""
    row = pd.DataFrame(
        [{
            "tenure_months": int(tenure_months),
            "monthly_charges": float(monthly_charges),
            "support_calls": int(support_calls),
            "contract_type": contract_type,
        }]
    )
    probability = float(MODEL.predict_proba(row[FEATURES])[0, 1])

    PREDICTION_COUNT["n"] += 1
    print(
        f"[predict] #{PREDICTION_COUNT['n']} "
        f"tenure={tenure_months} charges={monthly_charges} calls={support_calls} "
        f"contract={contract_type} -> {probability:.3f}"
    )

    if probability >= 0.65:
        verdict = "LIKELY TO LEAVE - worth a retention call"
    elif probability >= 0.35:
        verdict = "MIGHT LEAVE - keep an eye on this one"
    else:
        verdict = "LIKELY TO STAY"

    return (
        f"### {probability:.0%} chance of leaving\n"
        f"**{verdict}**\n\n"
        f"- Tenure: {int(tenure_months)} months\n"
        f"- Monthly bill: ${float(monthly_charges):.2f}\n"
        f"- Support calls last month: {int(support_calls)}\n"
        f"- Contract: {contract_type.replace('_', ' ')}\n\n"
        f"_Model version {MODEL_VERSION}. Predictions so far this session: {PREDICTION_COUNT['n']}._"
    )


EXAMPLES = [
    [6, 92.0, 4, "monthly"],     # new, expensive, annoyed -> high risk
    [48, 45.0, 0, "two_year"],   # loyal, cheap, quiet -> low risk
    [24, 70.0, 1, "one_year"],   # middle of the road
]

with gr.Blocks(title="Churn predictor - MDC-501 Lab") as demo:
    gr.Markdown(
        """
        # Will this customer leave?
        A real model, served in a browser, for free.

        This is the same idea as Lab 1's `/predict` -- except this one has a
        **contract**: the four boxes below cannot send a wrong type, and the
        contract type can only be one of three values. That is the difference
        between a notebook and a service.
        """
    )
    with gr.Row():
        with gr.Column():
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months as a customer)")
            charges = gr.Number(value=79.5, label="Monthly charges (USD)")
            calls = gr.Slider(0, 10, value=2, step=1, label="Support calls last month")
            contract = gr.Dropdown(CONTRACTS, value="monthly", label="Contract type")
            button = gr.Button("Predict", variant="primary")
        with gr.Column():
            output = gr.Markdown("Move the controls and click **Predict**.")
            gr.Examples(examples=EXAMPLES, inputs=[tenure, charges, calls, contract],
                        label="Try one of these")

    button.click(predict, inputs=[tenure, charges, calls, contract], outputs=output)

    gr.Markdown(
        """
        ---
        **Model status:** version `1.0.0-space`, trained on synthetic data,
        logistic regression, always-on. Every prediction is printed in the
        Space's own log -- that is your first baby step toward monitoring.

        *Built for MDC-501: From Notebook to Production.*
        """
    )


if __name__ == "__main__":
    # On Hugging Face Spaces, port 7860 is the expected door.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
