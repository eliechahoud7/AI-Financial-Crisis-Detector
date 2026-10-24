"""Chronological walk-forward splits with a purge gap (docs/decisions.md, D3 and D12).

Positions are row numbers in the S&P 500 trading calendar. The label for row i
uses closes up to row i + horizon, so with `purge` >= horizon:

- a fold for calendar year Y trains on eligible rows with position < v0 - purge,
  where v0 is the first trading row of Y; rows v0 - purge .. v0 - 1 are purged;
- every development row (train or validation) has position < t0 - purge, where
  t0 is the first test row, so no development label looks into the test period;
- the test set is every eligible row from t0 onward.

No rows are shuffled, and nothing is fitted here.
"""

import numpy as np
import pandas as pd


def walk_forward_splits(dates: pd.DatetimeIndex, eligible, validation_years: list[int], test_start: str,
                        purge: int, horizon: int) -> dict:
    dates = pd.DatetimeIndex(dates)
    eligible = np.asarray(eligible, dtype=bool)
    if len(dates) != len(eligible):
        raise ValueError("dates and eligible must have the same length")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("dates must be unique and sorted")
    if purge < horizon:
        raise ValueError(f"purge ({purge}) must be >= horizon ({horizon})")
    years = sorted(validation_years)
    if not years:
        raise ValueError("at least one validation year is required")
    test_ts = pd.Timestamp(test_start)
    if years[-1] >= test_ts.year:
        raise ValueError(f"validation years must end before the test start {test_start}")

    positions = np.arange(len(dates))
    test_positions = positions[dates >= test_ts]
    if len(test_positions) == 0:
        raise ValueError(f"no rows on or after test start {test_start}")
    t0 = int(test_positions[0])
    dev_limit = t0 - purge

    folds = []
    for year in years:
        in_year = positions[dates.year == year]
        if len(in_year) == 0:
            raise ValueError(f"no trading rows in validation year {year}")
        v0 = int(in_year[0])
        train = positions[eligible & (positions < v0 - purge)]
        purged = positions[eligible & (positions >= v0 - purge) & (positions < v0)]
        validation = positions[eligible & (dates.year == year) & (positions < dev_limit)]
        if len(train) == 0 or len(validation) == 0:
            raise ValueError(f"fold {year} has {len(train)} training and {len(validation)} validation rows")
        folds.append({"validation_year": year, "train": train, "purged": purged, "validation": validation,
                      "validation_rows_removed_before_test": int((eligible & (dates.year == year)
                                                                  & (positions >= dev_limit)).sum())})
    development = positions[eligible & (positions < dev_limit)]
    test_purged = positions[eligible & (positions >= dev_limit) & (positions < t0)]
    test = positions[eligible & (positions >= t0)]
    if len(test) == 0:
        raise ValueError("no eligible test rows")
    return {"folds": folds, "development": development, "test_purged": test_purged, "test": test,
            "first_test_position": t0, "purge": purge, "horizon": horizon}


def describe_splits(splits: dict, dates: pd.DatetimeIndex, target) -> dict:
    """Date ranges, row counts, and label-1 counts for each part (for metadata and reports)."""
    target = np.asarray(target)

    def part(idx):
        if len(idx) == 0:
            return {"rows": 0}
        return {"rows": int(len(idx)), "first_date": str(dates[idx[0]].date()), "last_date": str(dates[idx[-1]].date()),
                "label_1": int(np.nansum(target[idx])), "prevalence": round(float(np.nanmean(target[idx])), 6)}

    return {
        "folds": [{"validation_year": f["validation_year"], "train": part(f["train"]), "purged": part(f["purged"]),
                   "validation": part(f["validation"]),
                   "validation_rows_removed_before_test": f["validation_rows_removed_before_test"]}
                  for f in splits["folds"]],
        "development": part(splits["development"]), "test_purged": part(splits["test_purged"]),
        "test": part(splits["test"]), "purge": splits["purge"], "horizon": splits["horizon"],
    }
