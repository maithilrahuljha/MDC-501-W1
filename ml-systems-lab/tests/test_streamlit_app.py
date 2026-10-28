"""
WHAT: Checks the multi-lab Streamlit app: every page renders, every experiment runs, and the three Lab 1 failures are real.
HOW:  from the ml-systems-lab folder:  pytest tests -v
OK:   all tests pass; the app is clicked through headlessly in about 10 seconds.
"""

import pytest

pytest.importorskip("streamlit")  # the deployed app needs streamlit; skip if absent

from conftest import STREAMLIT, load_module  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = STREAMLIT / "app.py"
LABS = [
    "Start here",
    "Lab 1 - Break the notebook",
    "Lab 3 - Static vs dynamic (drift)",
    "Lab 4 - Batch vs online",
    "Lab 6 - Pipelines",
]


@pytest.fixture(scope="module")
def app_module():
    """Import the script WITHOUT launching the UI (it only builds UI under __main__)."""
    return load_module(APP_PATH, "streamlit_multilab_app")


@pytest.fixture(scope="module")
def model(app_module):
    trained, _ = app_module.train_churn()
    return trained


def test_sidebar_lists_every_lab():
    at = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
    assert not at.exception
    assert at.sidebar.radio[0].options == LABS


@pytest.mark.parametrize("lab", LABS[1:])
def test_each_lab_renders_without_an_exception(lab):
    at = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    at.sidebar.radio[0].set_value(lab).run()
    assert not at.exception, f"{lab} raised {[str(e.value) for e in at.exception]}"
    assert at.title, f"{lab} rendered no title"


def test_lab1_form_returns_a_probability():
    at = AppTest.from_file(str(APP_PATH), default_timeout=180).run()
    at.sidebar.radio[0].set_value("Lab 1 - Break the notebook").run()
    at.button[0].click().run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["Chance of leaving"]


def test_lab1_three_failures_are_real(app_module, model):
    """The app must reproduce Lab 1's exact error strings, not a friendly imitation."""
    errors = {
        name: app_module.naive_predict(model, case["body"])
        for name, case in app_module.BROKEN_CASES.items()
    }

    missing = next(v for k, v in errors.items() if k.startswith("Missing column"))
    wrong_type = next(v for k, v in errors.items() if k.startswith("Wrong data type"))
    unseen = next(v for k, v in errors.items() if k.startswith("Unseen category"))

    assert missing["ok"] is False and "columns are missing" in missing["error"]
    assert wrong_type["ok"] is False and "could not convert" in wrong_type["error"]
    assert unseen["ok"] is False and "unknown categories" in unseen["error"]


def test_a_valid_request_still_works(app_module, model):
    outcome = app_module.naive_predict(
        model,
        {"tenure_months": 12, "monthly_charges": 79.5, "support_calls": 2, "contract_type": "monthly"},
    )
    assert outcome["ok"] is True
    assert 0.0 <= outcome["probability"] <= 1.0


def test_drift_tab_separates_the_two_models(app_module):
    frame, first_below, drift_index = app_module.simulate_drift()
    tail = frame.tail(150)
    assert drift_index > 0
    assert first_below is not None, "the frozen model should visibly rot in this simulation"
    assert tail["dynamic_correct"].mean() > tail["static_correct"].mean() + 0.15


def test_benchmark_returns_sane_numbers(app_module, model):
    result = app_module.compare_batch_vs_single(model, rows=1000, sequential_rows=200)
    assert result["rows_batch"] == 1000
    assert result["batch_seconds"] > 0
    assert result["avg_latency_ms"] > 0
    assert result["slowdown"] > 1, "one-row-at-a-time must be slower than one batch call"


def test_drift_switch_changes_the_world(app_module):
    calm = app_module.make_training_data(n_rows=2000, seed=3, drift=0.0)
    stormy = app_module.make_training_data(n_rows=2000, seed=3, drift=1.0)
    assert stormy["monthly_charges"].mean() > calm["monthly_charges"].mean() + 10
    assert stormy["support_calls"].mean() > calm["support_calls"].mean()
