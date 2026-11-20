"""Model factory: one place that turns a model name (+ optional params) into a full Pipeline.

Every model is `preprocessor -> estimator`, so the same object can be cross-validated,
tuned, saved and used for prediction without any leakage.
"""
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from jocars.config import RANDOM_STATE
from jocars.features import build_preprocessor

# name -> (preprocessor kind, estimator class, default params)
_REGISTRY = {
    "ridge": ("linear", Ridge, {"alpha": 1.0}),
    "random_forest": ("tree", RandomForestRegressor,
                      {"n_estimators": 400, "min_samples_leaf": 2, "n_jobs": -1,
                       "random_state": RANDOM_STATE}),
    "hist_gb": ("tree", HistGradientBoostingRegressor,
                {"max_iter": 600, "learning_rate": 0.05, "random_state": RANDOM_STATE}),
    "xgboost": ("tree", XGBRegressor,
                {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
                 "colsample_bytree": 0.8, "n_jobs": -1, "random_state": RANDOM_STATE}),
    "lightgbm": ("tree", LGBMRegressor,
                 {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31,
                  "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                  "random_state": RANDOM_STATE, "verbose": -1}),
}
MODEL_NAMES = list(_REGISTRY)


def make_model(name: str, params: dict | None = None) -> Pipeline:
    """Build `preprocessor -> estimator`; `params` override the defaults."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose one of {MODEL_NAMES}")
    kind, estimator_cls, defaults = _REGISTRY[name]
    estimator = estimator_cls(**{**defaults, **(params or {})})
    return Pipeline([("prep", build_preprocessor(kind)), ("model", estimator)])
