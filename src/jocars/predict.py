"""Predict the price of one car described as a plain dict of raw listing fields.

Example:
    >>> from jocars.predict import load_model, predict_price
    >>> predict_price({"make": "Toyota", "model": "Camry", "year": "2019",
    ...                "mileage_km": 85000, "fuel_type": "Hybrid"}, load_model())
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from jocars.config import MODELS
from jocars.features import add_features, split_xy

# Raw input fields (same names as the CSV, without the price).
INPUT_FIELDS = [
    "make", "model", "year", "mileage_km", "mileage_text", "body_type", "condition",
    "transmission", "fuel_type", "engine_type", "color", "interior_color", "seats", "city",
    "feat_insurance", "feat_carcustoms", "feat_license", "feat_paintcondition",
    "feat_body_condition", "feat_battery_capacity", "feat_battery_range",
]


@lru_cache(maxsize=1)
def load_model(path=MODELS / "model.joblib") -> Pipeline:
    """Load the trained pipeline once (cached)."""
    return joblib.load(path)


def car_to_features(car: dict) -> pd.DataFrame:
    """Validate a car dict and turn it into the model's feature columns (one row)."""
    unknown = set(car) - set(INPUT_FIELDS)
    if unknown:
        raise ValueError(f"unknown field(s): {sorted(unknown)}; allowed: {INPUT_FIELDS}")
    if not car.get("year"):
        raise ValueError("'year' is required (e.g. '2019' or 'Older than 1970')")

    row = {field: car.get(field, np.nan) for field in INPUT_FIELDS}
    row["year"] = str(row["year"])
    if pd.notna(row["seats"]):
        row["seats"] = str(row["seats"])
    X, _ = split_xy(add_features(pd.DataFrame([row])))
    return X


def predict_price(car: dict, model: Pipeline | None = None) -> float:
    """Predicted listing price in JOD (the model predicts log-price; we undo the log)."""
    model = model if model is not None else load_model()
    return float(np.exp(model.predict(car_to_features(car))[0]))


def explain_price(car: dict, model: Pipeline | None = None) -> dict:
    """Why this price? SHAP values from LightGBM's built-in TreeSHAP (`pred_contrib=True`).

    Returns {"base_price": average-car price, "contributions": {feature: log-effect}}.
    Contributions are in log-price, so they multiply:  price = base * exp(sum of contributions);
    exp(c) - 1 is the % effect of one feature (e.g. 0.10 -> about +10.5%).
    """
    model = model if model is not None else load_model()
    X = car_to_features(car)
    contrib = model[-1].predict(model[:-1].transform(X), pred_contrib=True)[0]
    names = list(X.columns)  # ColumnTransformer keeps NUMERIC + CATEGORICAL order
    return {
        "base_price": float(np.exp(contrib[-1])),
        "contributions": dict(zip(names, map(float, contrib[:-1]))),
    }
