"""Tests for jocars.cleaning.clean: which listings are kept for training."""
import numpy as np

from conftest import make_row
from jocars.cleaning import clean


def test_keeps_a_normal_listing(frame):
    out = clean(frame(make_row()))
    assert len(out) == 1


def test_drops_fully_empty_rows(frame):
    empty = {k: np.nan for k in make_row()}
    out = clean(frame(make_row(), empty))
    assert len(out) == 1


def test_drops_rows_without_price(frame):
    out = clean(frame(make_row(), make_row(price_value_JOD=np.nan)))
    assert len(out) == 1


def test_drops_exact_duplicates(frame):
    out = clean(frame(make_row(), make_row()))
    assert len(out) == 1


def test_drops_placeholder_prices_at_or_above_200k(frame):
    out = clean(frame(make_row(price_value_JOD=999999.0), make_row(price_value_JOD=7_000_000.0)))
    assert len(out) == 0


def test_keeps_real_luxury_price_below_200k(frame):
    out = clean(frame(make_row(make="Mercedes Benz", model="G-Class", year="2025",
                               condition="New", price_value_JOD=165000.0)))
    assert len(out) == 1


def test_drops_call_me_prices_of_1000_or_less(frame):
    out = clean(frame(make_row(year="1995", price_value_JOD=1000.0),
                      make_row(year="1995", price_value_JOD=550.0)))
    assert len(out) == 0


def test_keeps_cheap_price_for_an_old_car(frame):
    out = clean(frame(make_row(year="1996", price_value_JOD=1400.0)))
    assert len(out) == 1


def test_drops_too_cheap_price_for_a_recent_car(frame):
    # a 2023 car listed at 1,400 JOD is a down payment, not a price
    out = clean(frame(make_row(year="2023", price_value_JOD=1400.0)))
    assert len(out) == 0


def test_does_not_modify_the_input(frame):
    df = frame(make_row(), make_row(price_value_JOD=np.nan))
    clean(df)
    assert len(df) == 2


def test_result_has_a_fresh_index(frame):
    out = clean(frame(make_row(price_value_JOD=np.nan), make_row()))
    assert list(out.index) == [0]
