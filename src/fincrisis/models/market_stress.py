"""Market-stress models and walk-forward prediction (docs/decisions.md, D12 and D13).

Rule baselines use a raw feature value as the score and fit nothing. Fitted
models are refitted on each fold's training rows only. Inputs are exactly
`market_features.FEATURES`; bookkeeping columns are never model inputs.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from fincrisis.features.market_features import FEATURES

BOOKKEEPING = ["position", "target", "eligible", "missing_features"]


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """The model input matrix: exactly FEATURES, in order, with no missing values."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = frame[FEATURES]
    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values; use eligible rows only")
    return X


def build_models(settings: dict, seed: int) -> dict:
    """name -> ('fit', estimator) or ('rule', feature column)."""
    lr, xgb = settings["logistic_regression"], settings["xgboost"]
    models = {"dummy": ("fit", DummyClassifier(strategy="prior"))}
    for column in settings["rule_baselines"]:
        if column not in FEATURES:
            raise ValueError(f"Rule baseline {column!r} is not a feature")
        models[f"rule_{column}"] = ("rule", column)
    models["logistic_regression"] = ("fit", Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], random_state=seed)),
    ]))
    models["xgboost"] = ("fit", XGBClassifier(**xgb, eval_metric="logloss", random_state=seed))
    return models


def is_probability(kind: str) -> bool:
    return kind == "fit"


def fit_predict(kind_and_model, X_train, y_train, X_eval):
    kind, model = kind_and_model
    if kind == "rule":
        return X_eval[model].to_numpy(dtype=float), None
    fitted = clone(model).fit(X_train, y_train)
    return fitted.predict_proba(X_eval)[:, 1], fitted


def walk_forward_predict(models: dict, frame: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """One prediction per validation row, from the model trained on that row's fold.

    `frame` must start at calendar position 0 so fold positions index it directly.
    """
    if not np.array_equal(frame["position"].to_numpy(), np.arange(len(frame))):
        raise ValueError("frame positions must run 0..n-1")
    rows = []
    for fold in folds:
        train, valid = fold["train"], fold["validation"]
        if set(train) & set(valid):
            raise ValueError(f"fold {fold['validation_year']}: training and validation overlap")
        train_rows, valid_rows = frame.iloc[train], frame.iloc[valid]
        X_train, X_valid = feature_matrix(train_rows), feature_matrix(valid_rows)
        block = pd.DataFrame({"position": valid, "date": valid_rows.index, "year": fold["validation_year"],
                              "label": valid_rows["target"].to_numpy()})
        for name, spec in models.items():
            block[name] = fit_predict(spec, X_train, train_rows["target"], X_valid)[0]
        rows.append(block)
    return pd.concat(rows, ignore_index=True)
