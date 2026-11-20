"""Tests for jocars.train: the final fit and the single test-set evaluation."""
import numpy as np

from conftest import make_row
from jocars.train import fit_and_evaluate


def _raw(frame, n=100):
    rng = np.random.default_rng(1)
    years = rng.integers(2008, 2024, n)
    rows = [make_row(year=str(y), price_value_JOD=float(4000 + 1200 * (y - 2008)),
                     mileage_km=float(1000 * i), make=["Kia", "Toyota"][i % 2])
            for i, y in enumerate(years)]
    return frame(*rows)


def test_fit_and_evaluate_reports_test_metrics_and_sizes(frame):
    model, report = fit_and_evaluate(_raw(frame), {"n_estimators": 50, "objective": "l1"})
    assert report["n_train"] == 80 and report["n_test"] == 20
    assert {"mae_jod", "median_ape", "within_20pct", "r2_log"} <= set(report["test"])


def test_fitted_model_predicts_log_prices_for_new_rows(frame):
    model, _ = fit_and_evaluate(_raw(frame), {"n_estimators": 50, "objective": "l1"})
    from jocars.features import add_features, split_xy
    X, _ = split_xy(add_features(frame(make_row(year="2020"))))
    price = np.exp(model.predict(X))[0]
    assert 4000 < price < 30000
