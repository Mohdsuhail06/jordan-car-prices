"""Tests for jocars.models.make_model: every model is a full Pipeline that fits and predicts."""
import numpy as np
import pytest

from conftest import make_row
from jocars.features import add_features, split_xy
from jocars.models import MODEL_NAMES, make_model


@pytest.fixture
def xy(frame):
    rng = np.random.default_rng(0)
    rows = [make_row(year=str(y), price_value_JOD=float(30000 - 1500 * (2024 - y)),
                     make=["Kia", "Toyota", "BYD"][i % 3])
            for i, y in enumerate(rng.integers(2010, 2024, 60))]
    return split_xy(add_features(frame(*rows)))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_fits_and_predicts_one_value_per_row(xy, name):
    X, y = xy
    model = make_model(name).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_custom_params_reach_the_estimator():
    model = make_model("lightgbm", {"n_estimators": 7})
    assert model[-1].get_params()["n_estimators"] == 7


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        make_model("deep_magic")
