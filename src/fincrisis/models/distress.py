"""Baseline pipelines for the company distress model (docs/decisions.md, D9).

Every data-dependent step lives inside a Pipeline, so it is refitted on each
training fold. Class weights are computed from the labels passed to each fit.
Label 1 is treated as the class to detect only because it is the minority
class; its real-world meaning is unverified.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier

from fincrisis.features.large_values import LargeValuesToMissing

# Treatments per model family, in order of preference when results are tied.
FAMILIES = {
    "dummy": ["dummy"],
    "logistic_regression": ["lr_quantile", "lr_large_as_missing", "lr_standard"],
    "xgboost": ["xgb_raw", "xgb_large_as_missing"],
}
# Ablations: reported, never selected.
DIAGNOSTICS = ["lr_quantile_drop24", "xgb_raw_drop24"]


class BalancedXGBClassifier(XGBClassifier):
    """XGBClassifier whose scale_pos_weight is count(0) / count(1) of each fit's labels."""

    def fit(self, X, y, **kwargs):
        labels = np.asarray(y)
        positives = int((labels == 1).sum())
        if positives == 0:
            raise ValueError("Training labels contain no examples of class 1")
        self.set_params(scale_pos_weight=float((labels == 0).sum() / positives))
        return super().fit(X, y, **kwargs)


def build_pipelines(settings: dict, large_columns: list[str], seed: int) -> dict[str, Pipeline]:
    """Return the named baseline configurations from the `company_distress` settings."""
    threshold = settings["large_value_threshold"]

    def logistic():
        lr = settings["logistic_regression"]
        return LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], class_weight="balanced", random_state=seed)

    def xgboost():
        return BalancedXGBClassifier(**settings["xgboost"], eval_metric="aucpr", random_state=seed)

    def quantile():
        return QuantileTransformer(output_distribution="normal", random_state=seed)

    def drop_large():
        return ColumnTransformer([("drop", "drop", list(large_columns))], remainder="passthrough")

    return {
        "dummy": Pipeline([("model", DummyClassifier(strategy="prior"))]),
        "lr_standard": Pipeline([("scale", StandardScaler()), ("model", logistic())]),
        "lr_quantile": Pipeline([("quantile", quantile()), ("model", logistic())]),
        "lr_large_as_missing": Pipeline([
            ("large", LargeValuesToMissing(threshold)),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", logistic()),
        ]),
        "xgb_raw": Pipeline([("model", xgboost())]),
        "xgb_large_as_missing": Pipeline([("large", LargeValuesToMissing(threshold)), ("model", xgboost())]),
        "lr_quantile_drop24": Pipeline([("drop", drop_large()), ("quantile", quantile()), ("model", logistic())]),
        "xgb_raw_drop24": Pipeline([("drop", drop_large()), ("model", xgboost())]),
    }


def large_value_columns(features, threshold: float) -> list[str]:
    """Columns with any value above `threshold` (label-free; call on training rows only)."""
    return [c for c in features.columns if (features[c] > threshold).any()]
