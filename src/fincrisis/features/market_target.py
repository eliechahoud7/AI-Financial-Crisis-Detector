"""Market-stress target (docs/decisions.md, D2).

For trading day t with S&P 500 close C_t and horizon H:

    DD_t = min over j in {t+1, ..., t+H} of ( C_j / max(C_t, ..., C_j) - 1 )
    y_t  = 1 if DD_t <= -threshold, else 0

The target intentionally uses future closes; it must never be used as a feature.
The last H rows have no complete future window and get NaN, never 0.
"""

import numpy as np
import pandas as pd

TOLERANCE = 1e-12  # "at least 5%": a decline equal to the threshold counts


def validate_close(close: pd.Series) -> None:
    """Prices must be sorted by unique date, present, finite, and positive. Nothing is interpolated."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise ValueError("close must be indexed by date")
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise ValueError("close dates must be unique and sorted")
    values = close.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} missing closes; prices are never interpolated")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("closes must be finite and positive")


def forward_min_decline(close: pd.Series, horizon: int) -> pd.Series:
    """DD_t for each row; NaN where fewer than `horizon` future rows exist."""
    validate_close(close)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    c = close.to_numpy(dtype=float)
    n = len(c)
    result = np.full(n, np.nan)
    complete = n - horizon  # rows 0 .. n-horizon-1 have a full window
    if complete > 0:
        running_max = c[:complete].copy()
        worst = np.full(complete, np.inf)
        for k in range(1, horizon + 1):
            future = c[k:k + complete]
            running_max = np.maximum(running_max, future)
            worst = np.minimum(worst, future / running_max - 1)
        result[:complete] = worst
    return pd.Series(result, index=close.index, name="forward_min_decline")


def stress_label(close: pd.Series, horizon: int, threshold: float) -> pd.Series:
    """y_t as float 1.0 / 0.0, NaN where the future window is incomplete."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    decline = forward_min_decline(close, horizon)
    label = np.where(decline.isna(), np.nan, (decline <= -threshold + TOLERANCE).astype(float))
    return pd.Series(label, index=close.index, name="target")
