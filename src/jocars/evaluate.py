"""Model evaluation: the model works in log(price), people think in JOD.

- mae_jod  : average absolute error in dinars ("on average we are off by X JOD")
- rmse_jod : like MAE but squares errors first, so big misses weigh more
- mape     : average error as a percentage of the true price
- median_ape   : the error of a TYPICAL car (half the cars are closer than this)
- within_20pct : share of cars predicted within +-20% of their listed price
- r2_log   : share of the variation in log(price) the model explains (1 = perfect, 0 = guessing the mean)
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from jocars.config import RANDOM_STATE


def regression_report(y_true_log, y_pred_log) -> dict[str, float]:
    """Compute metrics from log-price arrays; money metrics are converted back with exp()."""
    y_true_log, y_pred_log = np.asarray(y_true_log), np.asarray(y_pred_log)
    true, pred = np.exp(y_true_log), np.exp(y_pred_log)
    ape = np.abs(pred - true) / true
    return {
        "mae_jod": mean_absolute_error(true, pred),
        "rmse_jod": float(np.sqrt(mean_squared_error(true, pred))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "within_20pct": float(np.mean(ape < 0.20)),
        "r2_log": r2_score(y_true_log, y_pred_log),
    }


def cross_validate_model(model, X, y, folds: int = 5) -> dict[str, dict[str, float]]:
    """K-fold cross-validation on the TRAINING data only.

    The data is cut into `folds` parts; each part is predicted once by a model trained on the
    others. Returns {metric: {"mean": ..., "std": ...}} across folds.
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fit_idx, val_idx in kf.split(X):
        m = clone(model).fit(X.iloc[fit_idx], y.iloc[fit_idx])
        scores.append(regression_report(y.iloc[val_idx], m.predict(X.iloc[val_idx])))
    return {k: {"mean": float(np.mean([s[k] for s in scores])),
                "std": float(np.std([s[k] for s in scores]))} for k in scores[0]}
