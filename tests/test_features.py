"""Tests for jocars.features: new columns, X/y split, train/test split, preprocessing."""
import numpy as np
import pandas as pd
import pytest

from conftest import make_row
from jocars.features import (
    CATEGORICAL, NUMERIC, add_features, build_preprocessor, split_xy, train_test,
)


# ---------- add_features ----------

def test_car_age_is_data_year_minus_model_year(frame):
    out = add_features(frame(make_row(year="2018")))
    assert out.loc[0, "car_age"] == 6


def test_older_than_1970_becomes_an_age(frame):
    out = add_features(frame(make_row(year="Older than 1970")))
    assert out.loc[0, "car_age"] == 2024 - 1965


def test_next_year_models_have_age_zero_not_negative(frame):
    out = add_features(frame(make_row(year="2026", condition="New", mileage_km=0.0)))
    assert out.loc[0, "car_age"] == 0


def test_missing_mileage_with_200k_text_is_flagged_and_filled(frame):
    out = add_features(frame(make_row(mileage_km=np.nan, mileage_text="200,000")))
    assert out.loc[0, "mileage_200k_plus"] == 1
    assert out.loc[0, "mileage_km"] == 220_000


def test_normal_mileage_is_not_flagged(frame):
    out = add_features(frame(make_row()))
    assert out.loc[0, "mileage_200k_plus"] == 0
    assert out.loc[0, "mileage_km"] == 94999


def test_km_per_year_divides_mileage_by_age(frame):
    out = add_features(frame(make_row(year="2014", mileage_km=100_000.0)))
    assert out.loc[0, "km_per_year"] == pytest.approx(10_000)


def test_km_per_year_for_age_zero_does_not_divide_by_zero(frame):
    out = add_features(frame(make_row(year="2024", mileage_km=5_000.0)))
    assert out.loc[0, "km_per_year"] == pytest.approx(5_000)


def test_seats_more_than_9_becomes_10(frame):
    out = add_features(frame(make_row(seats="More than 9"), make_row(seats="7")))
    assert list(out["seats"]) == [10, 7]


def test_battery_columns_missing_means_none(frame):
    out = add_features(frame(make_row()))
    assert out.loc[0, "feat_battery_capacity"] == "None"
    assert out.loc[0, "feat_battery_range"] == "None"


def test_make_model_joins_brand_and_model(frame):
    out = add_features(frame(make_row(make="Kia", model="Other")))
    assert out.loc[0, "make_model"] == "Kia Other"


def test_redundant_columns_are_dropped(frame):
    out = add_features(frame(make_row()))
    for col in ["year", "mileage_text", "engine_type"]:
        assert col not in out.columns


def test_add_features_does_not_modify_the_input(frame):
    df = frame(make_row())
    add_features(df)
    assert "car_age" not in df.columns


# ---------- split_xy ----------

def test_split_xy_removes_target_from_features(frame):
    X, y = split_xy(add_features(frame(make_row(price_value_JOD=12000.0))))
    assert "price_value_JOD" not in X.columns
    assert y.iloc[0] == pytest.approx(np.log(12000.0))


def test_split_xy_keeps_exactly_the_model_columns(frame):
    X, _ = split_xy(add_features(frame(make_row())))
    assert set(X.columns) == set(NUMERIC) | set(CATEGORICAL)


# ---------- train_test ----------

def _many_rows(frame, n=200):
    rng = np.random.default_rng(0)
    rows = [make_row(price_value_JOD=float(p), model=f"M{i}")
            for i, p in enumerate(rng.integers(2000, 60000, n))]
    return frame(*rows)


def test_train_test_is_80_20(frame):
    train, test = train_test(_many_rows(frame))
    assert (len(train), len(test)) == (160, 40)


def test_train_test_is_reproducible(frame):
    df = _many_rows(frame)
    a, _ = train_test(df)
    b, _ = train_test(df)
    assert list(a.index) == list(b.index)


def test_train_and_test_do_not_overlap(frame):
    train, test = train_test(_many_rows(frame))
    assert set(train.index).isdisjoint(test.index)


# ---------- build_preprocessor ----------

@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_preprocessor_handles_unseen_categories_and_missing_values(frame, kind):
    train = add_features(frame(*[make_row(model=f"M{i % 3}") for i in range(30)]))
    test = add_features(frame(make_row(make="NeverSeen", model=None, city=None, mileage_km=np.nan)))
    X_train, _ = split_xy(train)
    X_test, _ = split_xy(test)

    pre = build_preprocessor(kind).fit(X_train)
    out = pre.transform(X_test)

    assert out.shape[0] == 1
    assert not np.isnan(np.asarray(out, dtype=float)).any()


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        build_preprocessor("magic")
