"""Read-only checks that today's code still reproduces the frozen results (docs/decisions.md, D23).

The frozen evaluation (D12, D13, D9) was run with the code of its day. If the
feature code, the target code, the saved models or the installed libraries
changed, the dashboard and the workflow would score differently from what was
evaluated, and nothing else would notice. These checks recompute, with the
current code, values that were saved by the frozen run and compare them:

- the 16 market features and the market-stress target, rebuilt from the frozen
  snapshot, against the frozen `dataset.csv`;
- the saved market-stress models' scores, and the rule scores, on the held-out
  test rows, against the frozen `test_predictions.csv`;
- the saved company-distress models' scores on the held-out test companies,
  against the frozen `test_predictions.csv`;
- the latest-day signal from `assess_market` (the dashboard and workflow path)
  against the same computation done from the frozen pieces directly.

Nothing is fitted or written. Models are loaded only after their checksum
check (`load_verified_model`); the inputs are loaded through the integrity-
checked loaders. Every model saw these rows during the frozen evaluation, so
this re-scores them; it does not evaluate anything again.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fincrisis.agents import tools
from fincrisis.dashboard import loaders
from fincrisis.data import taiwan_bankruptcy as tb
from fincrisis.features import market_features as mf
from fincrisis.features.market_target import stress_label
from fincrisis.qa.integrity import Check

# Saved prediction files store float32 model outputs as printed text, so a
# reproduced score can differ in the 8th significant digit. Anything larger is drift.
SCORE_TOLERANCE = 1e-6
FEATURE_TOLERANCE = 0.0  # features are float64 and written with full precision: exact


def _difference(group: str, name: str, expected, actual, tolerance: float) -> Check:
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    if expected.shape != actual.shape:
        return Check(group, name, "mismatch", f"shape {actual.shape}, frozen {expected.shape}")
    if not np.array_equal(np.isnan(expected), np.isnan(actual)):
        return Check(group, name, "mismatch", "missing values differ")
    gap = float(np.nanmax(np.abs(expected - actual))) if expected.size else 0.0
    if gap > tolerance:
        return Check(group, name, "mismatch", f"max difference {gap:.3g} > tolerance {tolerance:g}")
    return Check(group, name, "ok", f"max difference {gap:.3g}")


def _unavailable(group: str, error: Exception) -> list[Check]:
    return [Check(group, "inputs", "missing", f"{type(error).__name__}: {error}")]


def market_dataset_checks(root: Path, settings: dict) -> list[Check]:
    """Features and target rebuilt from the frozen snapshot vs the frozen dataset (D11, D12)."""
    group = "market dataset"
    paths, target = settings["paths"], settings["market_stress_target"]
    try:
        snapshot = loaders.market_snapshot(root, paths)
        dataset, _ = loaders.market_dataset(root, paths)
    except loaders.ArtifactError as error:
        return _unavailable(group, error)
    features, _ = mf.build_features(snapshot["spx"], snapshot["vix"])
    rebuilt = features.reindex(dataset.index)
    checks = [_difference(group, f"feature {name}", dataset[name], rebuilt[name], FEATURE_TOLERANCE)
              for name in mf.FEATURES]
    label = stress_label(snapshot["spx"], target["horizon_trading_days"], target["drawdown_threshold"])
    checks.append(_difference(group, "target", dataset["target"], label.reindex(dataset.index), 0.0))
    return checks


def market_model_checks(root: Path, settings: dict) -> list[Check]:
    """Saved market-stress models and rules re-scored on the frozen test rows (D13)."""
    group = "market models"
    paths = settings["paths"]
    try:
        results = loaders.market_stress_results(root, paths)
        dataset, _ = loaders.market_dataset(root, paths)
    except loaders.ArtifactError as error:
        return _unavailable(group, error)
    saved = results["test_predictions"]
    rows = dataset.loc[pd.DatetimeIndex(saved["date"]), mf.FEATURES]
    model_dir = Path(root) / paths["models"] / "market_stress"
    checks = []
    for name, kind in results["selection"]["models"].items():
        if kind == "rule":
            scores = rows[name.removeprefix("rule_")].to_numpy()
        else:
            scores = tools.load_verified_model(model_dir, name).predict_proba(rows)[:, 1]
        checks.append(_difference(group, f"{name} test scores", saved[name], scores, SCORE_TOLERANCE))
    return checks


def distress_model_checks(root: Path, settings: dict) -> list[Check]:
    """Saved company-distress models re-scored on the frozen test companies (D9)."""
    group = "distress models"
    paths = settings["paths"]
    try:
        results = loaders.distress_results(root, paths)
        X_test, y_test, row_ids = tb.load_processed(Path(root) / paths["interim_data"] / "taiwan_bankruptcy", "test")
    except (loaders.ArtifactError, tb.ValidationError, OSError) as error:
        return _unavailable(group, error)
    saved = results["test_predictions"]
    checks = [Check(group, "test rows", "ok" if saved["row_id"].tolist() == row_ids.tolist()
                    and saved["label"].tolist() == y_test.tolist() else "mismatch",
                    "" if saved["row_id"].tolist() == row_ids.tolist() else "row ids or labels differ")]
    if not checks[0].ok:
        return checks
    model_dir = Path(root) / paths["models"] / "taiwan_bankruptcy"
    for config in [c for c in saved.columns if c not in ("row_id", "label")]:
        scores = tools.load_verified_model(model_dir, config).predict_proba(X_test)[:, 1]
        checks.append(_difference(group, f"{config} test scores", saved[config], scores, SCORE_TOLERANCE))
    return checks


def latest_signal_checks(root: Path, settings: dict) -> list[Check]:
    """`assess_market` (dashboard and workflow) vs the frozen pieces combined directly, for the latest day."""
    group = "latest signal"
    paths = settings["paths"]
    try:
        signal = loaders.current_market_signal(root, paths, settings["agent"]["stale_after_days"])
        snapshot = loaders.market_snapshot(root, paths)
        results = loaders.market_stress_results(root, paths)
    except loaders.ArtifactError as error:
        return _unavailable(group, error)
    features, _ = mf.build_features(snapshot["spx"], snapshot["vix"])
    latest = features.iloc[[-1]]
    model_dir = Path(root) / paths["models"] / "market_stress"
    checks = [Check(group, "as-of date", "ok" if signal["as_of"] == str(latest.index[0].date()) else "mismatch",
                    f"{signal['as_of']} vs {latest.index[0].date()}")]
    for part in ("primary", "secondary"):
        described = signal.get(part)
        if not described:
            continue
        name = described["name"]
        if results["selection"]["models"][name] == "rule":
            expected = latest[name.removeprefix("rule_")].iloc[0]
        else:
            expected = tools.load_verified_model(model_dir, name).predict_proba(latest[mf.FEATURES])[:, 1][0]
        checks.append(_difference(group, f"{part} score ({name})", [expected], [described["score"]], 0.0))
        threshold_ok = described["threshold"] == results["thresholds"][name] and \
            described["elevated"] == bool(described["score"] >= described["threshold"])
        checks.append(Check(group, f"{part} threshold and signal ({name})", "ok" if threshold_ok else "mismatch"))
    return checks


def verify_all(root: Path = loaders.PROJECT_ROOT, settings: dict | None = None) -> list[Check]:
    settings = settings or loaders.load_settings()
    return (market_dataset_checks(root, settings) + market_model_checks(root, settings)
            + distress_model_checks(root, settings) + latest_signal_checks(root, settings))
