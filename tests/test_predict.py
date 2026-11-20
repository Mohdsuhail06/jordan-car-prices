"""Tests for jocars.predict: price for one car described as a plain dict."""
import numpy as np
import pytest

from conftest import make_row
from jocars.features import add_features, split_xy
from jocars.models import make_model
from jocars.predict import predict_price


@pytest.fixture(scope="module")
def model():
    import pandas as pd
    from conftest import RAW_COLUMNS
    rng = np.random.default_rng(2)
    rows = [make_row(year=str(y), price_value_JOD=float(4000 + 1200 * (y - 2008)),
                     make=["Kia", "Toyota"][i % 2])
            for i, y in enumerate(rng.integers(2008, 2024, 120))]
    X, y = split_xy(add_features(pd.DataFrame(rows, columns=RAW_COLUMNS)))
    return make_model("lightgbm", {"n_estimators": 100, "objective": "l1"}).fit(X, y)


def test_returns_a_price_in_jod(model):
    car = {"make": "Kia", "model": "Corolla", "year": "2020", "mileage_km": 50000}
    price = predict_price(car, model)
    assert isinstance(price, float)
    assert 10_000 < price < 20_000


def test_newer_car_is_predicted_more_expensive(model):
    old = predict_price({"make": "Kia", "year": "2010", "mileage_km": 50000}, model)
    new = predict_price({"make": "Kia", "year": "2022", "mileage_km": 50000}, model)
    assert new > old


def test_missing_fields_are_allowed(model):
    assert predict_price({"year": "2018"}, model) > 0


def test_explanation_adds_up_to_the_prediction(model):
    from jocars.predict import explain_price
    car = {"make": "Toyota", "year": "2015", "mileage_km": 120000}
    exp = explain_price(car, model)
    rebuilt = exp["base_price"] * np.exp(sum(exp["contributions"].values()))
    assert rebuilt == pytest.approx(predict_price(car, model), rel=1e-4)


def test_explanation_names_features_and_age_matters_most(model):
    from jocars.predict import explain_price
    exp = explain_price({"make": "Kia", "year": "2010", "mileage_km": 50000}, model)
    biggest = max(exp["contributions"], key=lambda k: abs(exp["contributions"][k]))
    assert biggest == "car_age"


def test_year_is_required(model):
    with pytest.raises(ValueError, match="year"):
        predict_price({"make": "Kia"}, model)


def test_unknown_field_names_are_rejected(model):
    with pytest.raises(ValueError, match="colour"):
        predict_price({"year": "2018", "colour": "Red"}, model)
