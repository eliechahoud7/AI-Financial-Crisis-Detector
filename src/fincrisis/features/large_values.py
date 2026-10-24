"""Pipeline step for the unexplained large values (docs/decisions.md, D4 and D9).

Values above the threshold are not assumed to be errors. This step only
expresses one hypothesis under test: that they are not on the same scale as
the other values, and are better treated as missing.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LargeValuesToMissing(TransformerMixin, BaseEstimator):
    """Replace values above `threshold` with NaN. Learns nothing from the data."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        values = np.array(X, dtype=float, copy=True)
        values[values > self.threshold] = np.nan
        return values
