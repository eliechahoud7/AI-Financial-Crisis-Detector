"""Daily market data snapshots: normalization and validation (docs/decisions.md, D6 and D10).

Only completed trading days are kept. Validation separates errors (the
snapshot must not be written) from warnings (recorded in metadata).
"""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
REQUIRED_COLUMNS = PRICE_COLUMNS + ["Adj Close", "Volume"]
MAX_START_LAG_DAYS = 10     # first row may start after holidays at the requested start
MAX_END_LAG_DAYS = 7        # last row must be close to the requested end (holidays, weekends)
GAP_WARNING_DAYS = 4        # calendar days between rows; weekend + one holiday = 4
MIN_WEEKDAY_COVERAGE = 0.95  # rows / weekdays in range (US market holidays are about 3.6%)


class MarketDataError(Exception):
    """The downloaded data failed validation; no snapshot should be written."""


def normalize_download(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a frame indexed by naive trading date with the standard columns, sorted by date."""
    if raw is None or raw.empty:
        raise MarketDataError("Download returned no rows")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise MarketDataError(f"Missing columns: {missing}; found {list(frame.columns)}")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame.index.name = "Date"
    return frame[REQUIRED_COLUMNS].sort_index()


def validate_series(frame: pd.DataFrame, ticker: str, start: date, end_inclusive: date) -> dict:
    """Check one daily series; return a summary with 'errors' and 'warnings' lists."""
    errors, warnings = [], []
    dates = frame.index
    duplicates = int(dates.duplicated().sum())
    if duplicates:
        errors.append(f"{duplicates} duplicate dates")
    if not dates.is_monotonic_increasing:
        errors.append("dates are not sorted")

    first, last = dates.min().date(), dates.max().date()
    if (first - start).days > MAX_START_LAG_DAYS:
        errors.append(f"first date {first} is more than {MAX_START_LAG_DAYS} days after requested start {start}")
    if first < start:
        errors.append(f"first date {first} is before requested start {start}")
    if last > end_inclusive:
        errors.append(f"last date {last} is after requested end {end_inclusive} (incomplete trading day?)")
    if (end_inclusive - last).days > MAX_END_LAG_DAYS:
        errors.append(f"last date {last} is more than {MAX_END_LAG_DAYS} days before requested end {end_inclusive}")

    missing = {c: int(frame[c].isna().sum()) for c in frame.columns}
    if missing["Close"]:
        errors.append(f"{missing['Close']} missing Close values")
    for c in frame.columns:
        if c != "Close" and missing[c]:
            warnings.append(f"{missing[c]} missing {c} values")

    nonpositive = {c: int((frame[c] <= 0).sum()) for c in PRICE_COLUMNS}
    if nonpositive["Close"]:
        errors.append(f"{nonpositive['Close']} non-positive Close values")
    for c in ("Open", "High", "Low"):
        if nonpositive[c]:
            warnings.append(f"{nonpositive[c]} non-positive {c} values")

    tolerance = 1e-9
    inconsistent = int(((frame["High"] < frame["Low"] - tolerance)
                        | (frame["Close"] > frame["High"] + tolerance)
                        | (frame["Close"] < frame["Low"] - tolerance)).sum())
    if inconsistent:
        warnings.append(f"{inconsistent} rows where Close is outside [Low, High] or High < Low")

    gaps = []
    steps = np.diff(dates.values).astype("timedelta64[D]").astype(int) if len(dates) > 1 else []
    for i, days in enumerate(steps):
        if days > GAP_WARNING_DAYS:
            gaps.append({"after": str(dates[i].date()), "before": str(dates[i + 1].date()), "calendar_days": int(days)})
    if gaps:
        warnings.append(f"{len(gaps)} gaps longer than {GAP_WARNING_DAYS} calendar days (listed in 'gaps')")

    weekdays = int(np.busday_count(first, last + timedelta(days=1)))
    coverage = len(frame) / weekdays if weekdays else 0.0
    if coverage < MIN_WEEKDAY_COVERAGE:
        errors.append(f"only {coverage:.1%} of weekdays have data (minimum {MIN_WEEKDAY_COVERAGE:.0%})")

    volume_zero = int((frame["Volume"] == 0).sum())
    return {
        "ticker": ticker, "rows": int(len(frame)), "columns": list(frame.columns),
        "first_date": str(first), "last_date": str(last), "missing_values": missing,
        "duplicate_dates": duplicates, "nonpositive_values": nonpositive, "ohlc_inconsistent_rows": inconsistent,
        "zero_volume_rows": volume_zero, "weekdays_in_range": weekdays, "weekday_coverage": round(coverage, 6),
        "gaps": gaps, "errors": errors, "warnings": warnings,
    }


def compare_calendars(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str, limit: int = 50) -> dict:
    """Trading dates present in one series but not the other (descriptive; no alignment is done here)."""
    only_a = a.index.difference(b.index)
    only_b = b.index.difference(a.index)
    return {
        f"dates_only_in_{name_a}": int(len(only_a)), f"dates_only_in_{name_b}": int(len(only_b)),
        f"examples_only_in_{name_a}": [str(d.date()) for d in only_a[:limit]],
        f"examples_only_in_{name_b}": [str(d.date()) for d in only_b[:limit]],
    }


def to_csv_text(frame: pd.DataFrame) -> str:
    """CSV with ISO dates; floats written so they parse back exactly."""
    out = frame.copy()
    out.index = out.index.strftime("%Y-%m-%d")
    text = out.to_csv(lineterminator="\n")
    parsed = pd.read_csv(io.StringIO(text), index_col="Date", float_precision="round_trip")
    same_values = np.array_equal(parsed.to_numpy(dtype=float), frame.to_numpy(dtype=float), equal_nan=True)
    if list(parsed.index) != list(out.index) or list(parsed.columns) != list(frame.columns) or not same_values:
        raise MarketDataError("CSV serialization does not round-trip exactly")
    return text
