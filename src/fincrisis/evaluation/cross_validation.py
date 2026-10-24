"""Repeated stratified cross-validation on the training set only (docs/decisions.md, D9).

Every fit receives only its fold's training rows; validation rows are used
only for prediction. Out-of-fold predictions feed threshold selection.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold

from fincrisis.evaluation.metrics import ranking_metrics, threshold_metrics


def make_folds(y, n_splits: int, n_repeats: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)), np.asarray(y)))


def cross_validate_configs(pipelines: dict, X: pd.DataFrame, y: pd.Series, folds, n_splits: int):
    """Return (per-fold scores, out-of-fold predictions [config -> (repeats, n)], warnings per config)."""
    n_repeats = len(folds) // n_splits
    scores, oof, caught = [], {}, {}
    for name, pipeline in pipelines.items():
        oof[name] = np.full((n_repeats, len(y)), np.nan)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            for k, (train_idx, valid_idx) in enumerate(folds):
                model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
                proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
                oof[name][k // n_splits, valid_idx] = proba
                y_valid = y.iloc[valid_idx].to_numpy()
                row = {"config": name, "repeat": k // n_splits, "fold": k % n_splits}
                row.update(ranking_metrics(y_valid, proba))
                row.update({f"{m}_at_0.5": v for m, v in threshold_metrics(y_valid, proba, 0.5).items()
                            if m in ("precision", "recall", "f1")})
                scores.append(row)
        caught[name] = sorted({f"{r.category.__name__}: {str(r.message).splitlines()[0][:160]}" for r in records})
        if np.isnan(oof[name]).any():
            raise RuntimeError(f"Out-of-fold predictions incomplete for {name}")
    return pd.DataFrame(scores), oof, caught


def paired_comparison(scores: pd.DataFrame, a: str, b: str, metric: str = "average_precision") -> dict:
    """Fold-by-fold difference a - b on the same folds."""
    pivot = scores.pivot_table(index=["repeat", "fold"], columns="config", values=metric)
    diff = (pivot[a] - pivot[b]).to_numpy()
    return {"a": a, "b": b, "metric": metric, "mean_difference": float(diff.mean()),
            "sd": float(diff.std(ddof=1)), "se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
            "folds_a_better": int((diff > 0).sum()), "n_folds": int(len(diff))}


def select_treatment(scores: pd.DataFrame, candidates: list[str], metric: str = "average_precision") -> dict:
    """Pick the first candidate (in preference order) within one standard error of the best mean.

    The standard error of the fold-by-fold difference treats repeated-CV folds as
    independent, which they are not; the rule is a conservative tie-break, not a test.
    """
    means = scores.groupby("config")[metric].mean()
    best = max(candidates, key=lambda c: means[c])
    result = {"candidates": list(candidates), "means": {c: float(means[c]) for c in candidates},
              "best_by_mean": best, "comparisons": []}
    if len(candidates) == 1:
        return {**result, "selected": best, "reason": "only candidate in this family"}
    for candidate in candidates:
        if candidate == best:
            reason = ("highest mean and first in preference order; no tie-break needed" if candidate == candidates[0]
                      else "highest mean; every earlier-preferred candidate was more than one standard error behind")
            return {**result, "selected": candidate, "reason": reason}
        comparison = paired_comparison(scores, best, candidate, metric)
        result["comparisons"].append(comparison)
        if comparison["mean_difference"] <= comparison["se"]:
            return {**result, "selected": candidate,
                    "reason": f"within one standard error of the highest mean ({best}); preferred by order"}
    raise AssertionError("unreachable")


def grouped_permutation_importance(pipeline, X: pd.DataFrame, y: pd.Series, folds, columns: list[str],
                                   rounds: int, seed: int) -> list[dict]:
    """Drop in average precision when `columns` are shuffled together in each validation fold."""
    rng = np.random.default_rng(seed)
    results = []
    for k, (train_idx, valid_idx) in enumerate(folds):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx].to_numpy()
        base = average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1])
        drops = []
        for _ in range(rounds):
            shuffled = X_valid.copy()
            order = rng.permutation(len(shuffled))
            shuffled[columns] = X_valid[columns].to_numpy()[order]
            drops.append(base - average_precision_score(y_valid, model.predict_proba(shuffled)[:, 1]))
        results.append({"fold": k, "base_average_precision": float(base), "mean_drop": float(np.mean(drops)),
                        "gain_share": _gain_share(model, list(X.columns), columns)})
    return results


def _gain_share(model, feature_names: list[str], columns: list[str]) -> float | None:
    """Share of XGBoost total gain from `columns`; None for other models."""
    estimator = model[-1]
    if not hasattr(estimator, "get_booster") or "drop" in model.named_steps:
        return None
    gains = estimator.get_booster().get_score(importance_type="total_gain")
    named = {}
    for key, value in gains.items():
        name = feature_names[int(key[1:])] if key not in feature_names and key[1:].isdigit() else key
        named[name] = value
    total = sum(named.values())
    return float(sum(v for k, v in named.items() if k in columns) / total) if total else None
