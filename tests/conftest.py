"""Shared test helpers: tiny hand-made DataFrames that look like the raw file."""
import numpy as np
import pandas as pd
import pytest

RAW_COLUMNS = [
    "price_value_JOD", "make", "model", "year", "mileage_km", "mileage_text", "body_type",
    "condition", "transmission", "fuel_type", "engine_type", "color", "interior_color", "seats",
    "city", "feat_insurance", "feat_carcustoms", "feat_license", "feat_paintcondition",
    "feat_body_condition", "feat_battery_capacity", "feat_battery_range",
]


def make_row(**overrides) -> dict:
    """A realistic, valid used-car listing; override any field you want to test."""
    row = {
        "price_value_JOD": 12000.0, "make": "Toyota", "model": "Corolla", "year": "2018",
        "mileage_km": 94999.0, "mileage_text": "90,000 - 99,999", "body_type": "Sedan",
        "condition": "Used", "transmission": "Automatic", "fuel_type": "Hybrid",
        "engine_type": "Hybrid", "color": "White", "interior_color": "Black", "seats": "5",
        "city": "Amman", "feat_insurance": "Compulsory Insurance", "feat_carcustoms": "With Customs",
        "feat_license": "Licensed", "feat_paintcondition": "Original Paint",
        "feat_body_condition": "Excellent with no defects",
        "feat_battery_capacity": np.nan, "feat_battery_range": np.nan,
    }
    row.update(overrides)
    return row


@pytest.fixture
def frame():
    """Build a DataFrame from rows: frame(make_row(), make_row(price_value_JOD=5000))."""
    def _frame(*rows):
        return pd.DataFrame(list(rows), columns=RAW_COLUMNS)
    return _frame
