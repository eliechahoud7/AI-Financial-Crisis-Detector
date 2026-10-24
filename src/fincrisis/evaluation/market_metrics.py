"""Metrics for time-ordered market-stress predictions (docs/decisions.md, D13).

Labels overlap across 20-session windows, so uncertainty is estimated with a
circular block bootstrap over the time-ordered rows rather than row by row.
Metrics that are undefined (one class only) are returned as None, never 0.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from fincrisis.evaluation.metrics import ranking_metrics


def score_metrics(y, score, is_probability: bool) -> dict:
    """Average precision and ROC-AUC; Brier and log-loss for probabilities only. None if undefined."""
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    if len(np.unique(y)) < 2:
        return {"average_precision": None, "roc_auc": None, "brier": None, "log_loss": None}
    if is_probability:
        return ranking_metrics(y, score)
    return {"average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)), "brier": None, "log_loss": None}


def per_year_metrics(years, y, score, is_probability: bool, min_positives: int) -> list[dict]:
    years, y, score = np.asarray(years), np.asarray(y), np.asarray(score, dtype=float)
    rows = []
    for year in np.unique(years):
        mask = years == year
        positives = int(y[mask].sum())
        m = score_metrics(y[mask], score[mask], is_probability)
        if m["average_precision"] is None:
            status = "undefined (one class only)"
        elif positives < min_positives:
            status = f"unreliable (fewer than {min_positives} label-1 rows)"
        else:
            status = "ok"
        rows.append({"year": int(year), "rows": int(mask.sum()), "label_1": positives,
                     "prevalence": float(y[mask].mean()), **m, "status": status})
    return rows


def mean_defined_average_precision(per_year: list[dict]) -> float | None:
    values = [r["average_precision"] for r in per_year if r["average_precision"] is not None]
    return float(np.mean(values)) if values else None


def circular_block_indices(n: int, block_length: int, n_resamples: int, seed: int):
    """Yield index arrays of length n made of contiguous (wrapping) blocks of the time-ordered rows."""
    if not 1 <= block_length <= n:
        raise ValueError(f"block_length must be between 1 and {n}")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    for _ in range(n_resamples):
        starts = rng.integers(0, n, size=blocks)
        yield ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]


def block_bootstrap_ci(y, scores: dict, block_length: int, n_resamples: int, seed: int, level: float = 0.95,
                       pairs: list[tuple[str, str]] = ()) -> dict:
    """Percentile intervals of average precision per score, and of paired differences, on shared resamples."""
    y = np.asarray(y)
    scores = {name: np.asarray(s, dtype=float) for name, s in scores.items()}
    samples = {name: [] for name in scores}
    diffs = {f"{a} - {b}": [] for a, b in pairs}
    skipped = 0
    for idx in circular_block_indices(len(y), block_length, n_resamples, seed):
        yy = y[idx]
        if yy.min() == yy.max():
            skipped += 1
            continue
        ap = {name: average_precision_score(yy, s[idx]) for name, s in scores.items()}
        for name in scores:
            samples[name].append(ap[name])
        for a, b in pairs:
            diffs[f"{a} - {b}"].append(ap[a] - ap[b])
    tail = (1 - level) / 2 * 100

    def interval(values):
        return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))] if values else None
    return {"block_length": block_length, "resamples": n_resamples, "skipped_one_class": skipped,
            "average_precision": {name: interval(v) for name, v in samples.items()},
            "differences": {k: {"mean": float(np.mean(v)) if v else None, "interval": interval(v)}
                            for k, v in diffs.items()}}
