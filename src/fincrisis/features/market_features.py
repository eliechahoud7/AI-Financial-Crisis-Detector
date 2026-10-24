"""Backward-looking market features (docs/decisions.md, D11 and D12).

S&P 500 features for date t use closes up to and including t (the prediction
is made after the close on t). VIX features for date t use only the VIX close
of the previous S&P 500 session (D11). Every rolling window requires a full
window of non-missing values; otherwise the feature is NaN. Nothing is filled.

r_t = ln(C_t / C_{t-1}); annualization uses sqrt(252).
"""

import numpy as np
import pandas as pd

from fincrisis.features.market_target import validate_close

ANNUALIZATION = np.sqrt(252)
RETURN_WINDOWS = (1, 5, 20)
VIX_CHANGE_WINDOWS = (5, 20)
HIGH_WINDOW = 252
DRAWDOWN_WINDOWS = (20, 60)
MA_WINDOW = 50
MA_SLOPE_LAG = 5
VIX_PERCENTILE_WINDOW = 252

FEATURE_DEFINITIONS = {
    "ret_1": "ln(C_t / C_{t-1})",
    "ret_5": "ln(C_t / C_{t-5})",
    "ret_20": "ln(C_t / C_{t-20})",
    "rv_5": "sample std (ddof=1) of the last 5 daily log returns r_{t-4..t}, x sqrt(252)",
    "rv_ratio_20_60": "rv_20 / rv_60, each the annualized sample std of the last 20 / 60 daily log returns",
    "downside_vol_20": "sqrt(mean(min(r, 0)^2)) over the last 20 daily log returns, x sqrt(252)",
    "drawdown_252": "C_t / max(C_{t-251..t}) - 1 (full 252-close window required)",
    "max_drawdown_20": "min over k in the last 20 closes of C_k / max(closes from window start to k) - 1",
    "max_drawdown_60": "same as max_drawdown_20 over the last 60 closes",
    "dist_ma_50": "C_t / SMA50_t - 1, SMA50 = mean of the last 50 closes",
    "ma_50_slope": "SMA50_t / SMA50_{t-5} - 1",
    "vix_lag1": "VIX close on the previous S&P 500 session (D11)",
    "vix_change_5": "ln(vix_lag1_t / vix_lag1_{t-5}), lags counted in S&P 500 sessions",
    "vix_change_20": "ln(vix_lag1_t / vix_lag1_{t-20})",
    "vix_pct_252": "share of vix_lag1 values in the last 252 sessions (including t) that are <= vix_lag1_t",
    "vix_rv_ratio": "vix_lag1 / (100 x rv_20 of the previous session); both known before the close on t",
}
FEATURES = list(FEATURE_DEFINITIONS)


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def realized_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std(ddof=1) * ANNUALIZATION


def downside_vol(returns: pd.Series, window: int) -> pd.Series:
    squared = np.minimum(returns, 0.0) ** 2
    return np.sqrt(squared.rolling(window, min_periods=window).mean()) * ANNUALIZATION


def drawdown_from_high(close: pd.Series, window: int) -> pd.Series:
    return close / close.rolling(window, min_periods=window).max() - 1


def max_drawdown(close: pd.Series, window: int) -> pd.Series:
    def worst(values: np.ndarray) -> float:
        return float((values / np.maximum.accumulate(values) - 1).min())
    return close.rolling(window, min_periods=window).apply(worst, raw=True)


def moving_average(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Share of the trailing `window` values (including the current one) <= the current value."""
    def share(values: np.ndarray) -> float:
        return np.nan if np.isnan(values).any() else float((values <= values[-1]).mean())
    return series.rolling(window, min_periods=window).apply(share, raw=True)


def align_vix_previous_session(spx_dates: pd.DatetimeIndex, vix_close: pd.Series) -> tuple[pd.Series, dict]:
    """D11: align VIX to S&P 500 dates (VIX-only dates are ignored), then lag by one S&P session."""
    if vix_close.index.has_duplicates:
        raise ValueError("VIX dates must be unique")
    aligned = vix_close.reindex(spx_dates)
    lagged = aligned.shift(1).rename("vix_lag1")
    vix_only = vix_close.index.difference(spx_dates)
    spx_without_vix = spx_dates[aligned.isna().to_numpy()]
    report = {
        "vix_only_dates_ignored": [str(d.date()) for d in vix_only],
        "spx_dates_without_vix_close": [str(d.date()) for d in spx_without_vix],
        "vix_lag1_missing": int(lagged.isna().sum()),
        "vix_lag1_missing_reason": "first S&P 500 date has no previous session; any other missing value means "
                                   "the previous S&P 500 session had no VIX close",
    }
    return lagged, report


def build_features(spx_close: pd.Series, vix_close: pd.Series) -> tuple[pd.DataFrame, dict]:
    """All features indexed by S&P 500 date, plus the VIX alignment report."""
    validate_close(spx_close)
    r = log_returns(spx_close)
    features = pd.DataFrame(index=spx_close.index)
    for k in RETURN_WINDOWS:
        features[f"ret_{k}"] = np.log(spx_close / spx_close.shift(k))
    features["rv_5"] = realized_vol(r, 5)
    rv_20 = realized_vol(r, 20)
    features["rv_ratio_20_60"] = rv_20 / realized_vol(r, 60)
    features["downside_vol_20"] = downside_vol(r, 20)
    features["drawdown_252"] = drawdown_from_high(spx_close, HIGH_WINDOW)
    for w in DRAWDOWN_WINDOWS:
        features[f"max_drawdown_{w}"] = max_drawdown(spx_close, w)
    sma = moving_average(spx_close, MA_WINDOW)
    features["dist_ma_50"] = spx_close / sma - 1
    features["ma_50_slope"] = sma / sma.shift(MA_SLOPE_LAG) - 1

    vix_lag1, report = align_vix_previous_session(spx_close.index, vix_close)
    if (vix_lag1.dropna() <= 0).any():
        raise ValueError("VIX closes must be positive")
    features["vix_lag1"] = vix_lag1
    for k in VIX_CHANGE_WINDOWS:
        features[f"vix_change_{k}"] = np.log(vix_lag1 / vix_lag1.shift(k))
    features["vix_pct_252"] = rolling_percentile(vix_lag1, VIX_PERCENTILE_WINDOW)
    features["vix_rv_ratio"] = vix_lag1 / (100 * rv_20.shift(1))
    return features[FEATURES], report


def missingness(features: pd.DataFrame) -> dict:
    """Per feature: leading (warm-up) NaNs versus NaNs after the first valid value."""
    result = {}
    for col in features.columns:
        isna = features[col].isna().to_numpy()
        leading = int(np.argmax(~isna)) if (~isna).any() else len(isna)
        result[col] = {"missing": int(isna.sum()), "warm_up": leading, "after_warm_up": int(isna[leading:].sum())}
    return result
