"""Feature engineering: turn cleaned listings into model-ready columns.

Two layers, on purpose:
1. `add_features`  - row-by-row rules that need no statistics (safe to run before the split).
2. `build_preprocessor` - steps that LEARN from data (medians, category lists). They live inside
   a scikit-learn Pipeline, so they are fitted on the training set only -> no data leakage.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from jocars.cleaning import parse_year
from jocars.config import DATA_YEAR, RANDOM_STATE, TARGET

MILEAGE_200K_PLUS = 220_000  # stand-in value for the "200,000+" bracket
MIN_CATEGORY_COUNT = 10      # rarer categories are grouped as "infrequent"

NUMERIC = ["car_age", "mileage_km", "km_per_year", "seats", "mileage_200k_plus"]
CATEGORICAL = [
    "make", "make_model", "body_type", "condition", "transmission", "fuel_type", "color",
    "interior_color", "city", "feat_insurance", "feat_carcustoms", "feat_license",
    "feat_paintcondition", "feat_body_condition", "feat_battery_capacity", "feat_battery_range",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with engineered columns and redundant raw columns removed."""
    out = df.copy()

    out["car_age"] = (DATA_YEAR - parse_year(out["year"])).clip(lower=0)

    # In the raw data, a missing mileage_km always comes with mileage_text "200,000" (= 200k+ km).
    plus = out["mileage_km"].isna() & out["mileage_text"].eq("200,000")
    out["mileage_200k_plus"] = plus.astype(int)
    out.loc[plus, "mileage_km"] = MILEAGE_200K_PLUS
    out["km_per_year"] = out["mileage_km"] / out["car_age"].clip(lower=1)

    out["seats"] = pd.to_numeric(out["seats"].replace("More than 9", "10"), errors="coerce")

    # Battery info is empty for non-electric cars: the gap itself is information.
    for col in ["feat_battery_capacity", "feat_battery_range"]:
        out[col] = out[col].astype("object").fillna("None")

    make = out["make"].astype("object").fillna("Unknown")
    model = out["model"].astype("object").fillna("Unknown")
    out["make_model"] = make + " " + model

    return out.drop(columns=["year", "mileage_text", "engine_type"])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Features X and target y = log(price). y is None when there is no price column."""
    X = df[NUMERIC + CATEGORICAL]
    y = np.log(df[TARGET]) if TARGET in df.columns else None
    return X, y


def train_test(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproducible 80/20 split, stratified on price deciles so both sides cover all price levels."""
    bins = pd.qcut(df[TARGET], q=10, labels=False, duplicates="drop")
    return train_test_split(df, test_size=test_size, random_state=RANDOM_STATE, stratify=bins)


def build_preprocessor(kind: str) -> ColumnTransformer:
    """Learned preprocessing.

    kind="linear": impute, scale numbers, one-hot encode categories (linear models need this).
    kind="tree":   impute, ordinal-encode categories (trees split on numbers; no scaling needed).
    """
    fill_missing = SimpleImputer(strategy="constant", fill_value="Missing")
    if kind == "linear":
        num = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        cat = make_pipeline(fill_missing, OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT,
            sparse_output=False))
    elif kind == "tree":
        num = SimpleImputer(strategy="median")
        cat = make_pipeline(fill_missing, OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            min_frequency=MIN_CATEGORY_COUNT))
    else:
        raise ValueError(f"unknown preprocessor kind: {kind!r} (use 'linear' or 'tree')")

    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])
