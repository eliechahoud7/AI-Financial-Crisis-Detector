"""Metrics for imbalanced binary classification (docs/decisions.md, D9).

Label 1 is the class to detect (the minority class). Its real-world meaning
is unverified.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)


def threshold_metrics(y, proba, threshold: float) -> dict:
    y, flagged = np.asarray(y), np.asarray(proba) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold), "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def ranking_metrics(y, proba) -> dict:
    proba = np.asarray(proba, dtype=float)
    return {
        "average_precision": float(average_precision_score(y, proba)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
    }


def select_threshold_max_f1(y, proba) -> float:
    """Threshold maximizing F1 on the given predictions (use training out-of-fold predictions only)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(thresholds[int(np.argmax(f1))])


def threshold_table(y, proba, thresholds) -> list[dict]:
    return [threshold_metrics(y, proba, t) for t in thresholds]


def calibration_table(y, proba, bins: int = 10) -> list[dict]:
    y, proba = np.asarray(y), np.asarray(proba, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_predicted": float(proba[mask].mean()), "observed_rate": float(y[mask].mean())})
    return rows


def stratified_bootstrap_indices(y, n_resamples: int, seed: int):
    """Yield index arrays resampled with replacement within each class, so both classes always appear."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == label) for label in (0, 1)]
    for _ in range(n_resamples):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def bootstrap_ci(y, proba, threshold: float, n_resamples: int, seed: int, level: float = 0.95) -> dict:
    """Percentile intervals for ranking and threshold metrics."""
    y, proba = np.asarray(y), np.asarray(proba, dtype=float)
    names = ["average_precision", "roc_auc", "precision", "recall", "f1"]
    samples = {name: [] for name in names}
    for idx in stratified_bootstrap_indices(y, n_resamples, seed):
        ranked = {"average_precision": average_precision_score(y[idx], proba[idx]),
                  "roc_auc": roc_auc_score(y[idx], proba[idx])}
        ranked.update(threshold_metrics(y[idx], proba[idx], threshold))
        for name in names:
            samples[name].append(ranked[name])
    tail = (1 - level) / 2 * 100
    return {name: [float(np.percentile(v, tail)), float(np.percentile(v, 100 - tail))] for name, v in samples.items()}


def paired_bootstrap_diff(y, proba_a, proba_b, n_resamples: int, seed: int, level: float = 0.95) -> dict:
    """Interval for average_precision(a) - average_precision(b) on the same resamples."""
    y, a, b = np.asarray(y), np.asarray(proba_a, dtype=float), np.asarray(proba_b, dtype=float)
    diffs = [average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i])
             for i in stratified_bootstrap_indices(y, n_resamples, seed)]
    tail = (1 - level) / 2 * 100
    return {"mean_difference": float(np.mean(diffs)),
            "interval": [float(np.percentile(diffs, tail)), float(np.percentile(diffs, 100 - tail))]}
