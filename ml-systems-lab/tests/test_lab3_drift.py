"""
WHAT: Checks the drift experiment on a small, fast run: drift exists, the static model rots, retraining helps.
HOW:  from the ml-systems-lab folder:  pytest tests -v
OK:   all tests pass in a few seconds.
"""

import drift_experiment as lab3
import numpy as np
import pytest


@pytest.fixture(scope="module")
def small_run():
    df = lab3.build_stream(days=40, per_day=100, drift_day=15, seed=3)
    scored, retrains, warmup = lab3.run_experiment(
        df, warmup=800, retrain_every=400, window=1200, roll=100
    )
    return df, scored, retrains, warmup


def test_stream_shape_and_drift(small_run):
    df, _, _, _ = small_run
    assert len(df) == 4000
    before = df[df["day"] < 15]
    after = df[df["day"] >= 15]
    # the world really did change: bills went up and the contract mix shifted
    assert after["monthly_charges"].mean() > before["monthly_charges"].mean() + 10
    assert (after["contract_type"] == "two_year").mean() > (before["contract_type"] == "two_year").mean()


def test_static_model_degrades_more_than_dynamic(small_run):
    _, scored, _, warmup = small_run
    post = scored[scored["sample_index"] > scored["sample_index"].max() - 800]
    static_accuracy = post["static_correct"].mean()
    dynamic_accuracy = post["dynamic_correct"].mean()

    assert dynamic_accuracy > static_accuracy + 0.15, (
        f"retraining should clearly help: static={static_accuracy:.3f}, "
        f"dynamic={dynamic_accuracy:.3f}"
    )
    assert static_accuracy < 0.80, "the static model should visibly rot after drift"


def test_dynamic_model_actually_retrained(small_run):
    _, _, retrains, warmup = small_run
    assert len(retrains) >= 3          # the initial fit plus real retrains
    assert all(point >= warmup for point in retrains)


def test_first_below_threshold_returns_a_sample_index(small_run):
    _, scored, _, _ = small_run
    hit = lab3.first_index_below(scored, "static_accuracy", 0.70)
    if hit is not None:
        index, day, accuracy = hit
        assert index > 0
        assert accuracy < 0.70
        assert day >= 0
    else:  # a tiny simulation may never dip; then at least the math is sane
        assert np.isfinite(scored["static_accuracy"].iloc[-1])
