"""Final training: fit the chosen model on the whole training set and score the test set ONCE.

Run:  python -m jocars.train
Reads  models/best_params.json  (written by notebooks/03_models.ipynb)
Writes models/model.joblib and reports/test_metrics.json
"""
import json

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from jocars.cleaning import clean
from jocars.config import MODELS, ROOT
from jocars.data import load_raw
from jocars.evaluate import regression_report
from jocars.features import add_features, split_xy, train_test
from jocars.models import make_model

MODEL_NAME = "lightgbm"


def fit_and_evaluate(raw: pd.DataFrame, params: dict) -> tuple[Pipeline, dict]:
    """Clean -> features -> split -> fit on train -> metrics on the untouched test set."""
    train, test = train_test(add_features(clean(raw)))
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)

    model = make_model(MODEL_NAME, params).fit(X_train, y_train)
    report = {
        "model": MODEL_NAME,
        "params": params,
        "n_train": len(train),
        "n_test": len(test),
        "test": regression_report(y_test, model.predict(X_test)),
    }
    return model, report


def main() -> None:
    params = json.loads((MODELS / "best_params.json").read_text())
    model, report = fit_and_evaluate(load_raw(), params)

    joblib.dump(model, MODELS / "model.joblib")
    (ROOT / "reports" / "test_metrics.json").write_text(json.dumps(report, indent=2))

    t = report["test"]
    print(f"trained on {report['n_train']:,} listings, tested on {report['n_test']:,} unseen listings")
    print(f"  typical error (median) : {t['median_ape']:.1%}")
    print(f"  within +-20% of price  : {t['within_20pct']:.0%}")
    print(f"  MAE                    : {t['mae_jod']:,.0f} JOD")
    print(f"  MAPE                   : {t['mape']:.1%}")
    print(f"  R2 (log price)         : {t['r2_log']:.3f}")
    print(f"saved -> {MODELS / 'model.joblib'}")


if __name__ == "__main__":
    main()
