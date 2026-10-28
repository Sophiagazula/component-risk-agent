"""Forecast model checks: no leakage, exported trees match LightGBM, sane outputs."""
import copy
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from model import DATA, ROOT, H, QUANTILES, FIRST_ORIGIN, ForecastData, walk_predict, widen

MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def fd():
    return ForecastData.from_csv()


def test_features_ignore_the_future(fd):
    """Scrambling every month after the origin must not change any feature."""
    for i in (FIRST_ORIGIN, 24, fd.T - 1 - H):
        before = [fd.features(i, h, p) for h in range(1, H + 1) for p in range(len(fd.pids))]
        future_scrambled = copy.copy(fd)
        future_scrambled.Y = fd.Y.copy()
        future_scrambled.Y[i + 1:] = np.random.default_rng(0).uniform(0, 1e5, fd.Y[i + 1:].shape)
        after = [future_scrambled.features(i, h, p) for h in range(1, H + 1) for p in range(len(fd.pids))]
        assert before == after, f"feature at origin {i} reads data after the origin"


def test_training_rows_only_use_observed_targets(fd):
    for o in (20, 30, fd.T - 1):
        fd.rows_for(range(FIRST_ORIGIN, o + 1), o)
        assert max(fd.last_targets) <= o


def test_browser_trees_match_lightgbm():
    """The exported tree file (what the browser runs) reproduces each saved LightGBM model."""
    export = json.loads((MODELS / "forecast_model.json").read_text())
    fixture = json.loads((MODELS / "parity_fixture.json").read_text())
    X = pd.DataFrame(fixture["X"], columns=fixture["features"])
    X["product"] = X["product"].astype(int)
    for q, trees in zip(QUANTILES, export["models"]):
        booster = lgb.Booster(model_file=str(MODELS / f"quantile_q{int(q * 100):02d}.txt"))
        ref = booster.predict(X)
        mine = np.array([walk_predict(trees, row) for row in X.values])
        assert np.abs(ref - mine).max() < 1e-5


def test_forecast_quantiles_are_ordered_and_non_negative():
    fc = pd.read_csv(DATA / "forecast.csv")
    q = fc[[f"q{int(x * 100):02d}" for x in QUANTILES]].values
    assert (q >= 0).all()
    assert (np.diff(q, axis=1) >= -1e-9).all()


def test_widen_keeps_median_and_scales_spread():
    logq = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    out = widen(logq, 2.0)
    assert out[0, 3] == 4.0
    assert np.allclose(out[0] - 4.0, (logq[0] - 4.0) * 2.0)


def test_backtest_beats_naive_baselines():
    s = json.loads((DATA / "backtest_summary.json").read_text())["overall"]
    assert s["wape_model"] < min(s["wape_seasonal_naive"], s["wape_moving_avg"])
