"""Tests for jocars.evaluate: metrics in JOD and cross-validation."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from jocars.evaluate import cross_validate_model, regression_report


def test_perfect_predictions_have_zero_error():
    y = np.log([5000.0, 10000.0, 20000.0])
    r = regression_report(y, y)
    assert r["mae_jod"] == pytest.approx(0)
    assert r["rmse_jod"] == pytest.approx(0)
    assert r["mape"] == pytest.approx(0)
    assert r["r2_log"] == pytest.approx(1)


def test_errors_are_measured_in_jod_not_log():
    # true 10,000 and 20,000; predicted 11,000 and 18,000 -> errors 1,000 and 2,000 JOD
    y_true = np.log([10000.0, 20000.0])
    y_pred = np.log([11000.0, 18000.0])
    r = regression_report(y_true, y_pred)
    assert r["mae_jod"] == pytest.approx(1500)
    assert r["rmse_jod"] == pytest.approx(np.sqrt((1000**2 + 2000**2) / 2))
    assert r["mape"] == pytest.approx(0.10)  # 10% off in both cases


def test_median_error_and_share_within_20_percent_resist_a_few_wild_misses():
    # three cars predicted 10% off, one predicted 10x too high (a fake price in the data)
    y_true = np.log([10000.0, 10000.0, 10000.0, 1000.0])
    y_pred = np.log([11000.0, 9000.0, 11000.0, 10000.0])
    r = regression_report(y_true, y_pred)
    assert r["median_ape"] == pytest.approx(0.10)
    assert r["within_20pct"] == pytest.approx(0.75)


def test_r2_on_log_scale_is_zero_for_mean_prediction():
    y = np.log([5000.0, 10000.0, 20000.0])
    r = regression_report(y, np.full(3, y.mean()))
    assert r["r2_log"] == pytest.approx(0)


def _linear_data(n=100):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.uniform(0, 1, n)})
    y = pd.Series(8 + 2 * X["x"])  # a log-price perfectly explained by x
    return X, y


def test_cross_validation_returns_mean_and_std_per_metric():
    X, y = _linear_data()
    res = cross_validate_model(LinearRegression(), X, y, folds=5)
    assert set(res) >= {"mae_jod", "rmse_jod", "mape", "r2_log"}
    assert set(res["r2_log"]) == {"mean", "std"}


def test_cross_validation_tells_a_good_model_from_a_dummy():
    X, y = _linear_data()
    good = cross_validate_model(LinearRegression(), X, y, folds=5)
    dummy = cross_validate_model(DummyRegressor(strategy="median"), X, y, folds=5)
    assert good["r2_log"]["mean"] > 0.99
    assert dummy["r2_log"]["mean"] < 0.1
