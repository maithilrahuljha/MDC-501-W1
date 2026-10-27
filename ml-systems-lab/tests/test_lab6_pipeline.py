"""
WHAT: Checks the Lab 6 pipeline steps and the deployable Hugging Face app, without touching any cloud account.
HOW:  from the ml-systems-lab folder:  pytest tests -v
OK:   all tests pass; no credentials, no network, no cost.
"""


import pytest
from conftest import LAB5, LAB6, load_module

pipeline = load_module(LAB6 / "pipeline.py", "lab6_pipeline")


def test_data_is_synthetic_and_has_the_columns_we_document():
    data = pipeline.make_data(n_rows=300, seed=5)
    assert len(data) == 300
    assert set(pipeline.FEATURES).issubset(data.columns)
    assert set(data["contract_type"].unique()).issubset(set(pipeline.CONTRACTS))
    assert 0.01 < data["churn"].mean() < 0.99


def test_train_returns_metrics_that_look_like_a_real_model():
    data = pipeline.make_data(n_rows=1200, seed=5)
    model, metrics, params = pipeline.train(data, seed=5, c_value=0.5)

    assert 0.6 < metrics["accuracy"] <= 1.0
    assert 0.5 < metrics["roc_auc"] <= 1.0
    assert params["C"] == 0.5
    assert metrics["train_rows"] + metrics["test_rows"] == 1200
    assert hasattr(model, "predict_proba")


def test_vertex_mode_exits_cleanly_without_credentials(monkeypatch, capsys):
    """The friendly-message requirement: no traceback, no crash, exit code 0."""
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GCP_PROJECT", raising=False)
    monkeypatch.delenv("GCS_BUCKET", raising=False)

    import sys

    # main() reads sys.argv, so drive it exactly the way a student would.
    monkeypatch.setattr(sys, "argv", ["pipeline.py", "--mode", "vertex"])
    exit_code = pipeline.main()

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert "VERTEX MODE NEEDS A FEW THINGS FIRST" in captured
    assert "--mode local" in captured


def test_vertex_train_mode_writes_a_model(tmp_path):
    import sys

    argv = ["pipeline.py", "--mode", "vertex-train", "--model-dir", str(tmp_path), "--rows", "400"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        assert pipeline.main() == 0
    finally:
        sys.argv = old_argv

    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_space_app_answers_one_customer():
    pytest.importorskip("gradio")  # the deployable app needs gradio; skip if absent
    space = load_module(LAB5 / "app.py", "lab5_space_app")
    high_risk = space.predict(4, 120.0, 5, "monthly")
    low_risk = space.predict(60, 30.0, 0, "two_year")
    assert "chance of leaving" in high_risk
    assert "chance of leaving" in low_risk
    assert "LIKELY TO LEAVE" in high_risk
    assert "LIKELY TO STAY" in low_risk
