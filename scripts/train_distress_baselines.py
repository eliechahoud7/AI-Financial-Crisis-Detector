"""Baseline company distress models on the Taiwanese Bankruptcy data (docs/decisions.md, D9).

Usage:
    python scripts/train_distress_baselines.py --stage cv       # training set only
    python scripts/train_distress_baselines.py --stage final    # after review: one test evaluation

Options: --force (overwrite outputs of the chosen stage), --input-dir, --output-dir,
--artifacts-dir, --smoke (tiny settings for tests).

Stage "cv" loads training rows only, runs repeated stratified CV, freezes the
selected treatment and threshold per model family in selection.json, and never
reads test labels. Stage "final" requires selection.json, fits each selected
configuration on the full training set, and evaluates the held-out test set once.

Label 1 is the class to detect because it is the minority class. The meanings of
labels 0 and 1 are not documented by UCI and remain unverified.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml

from fincrisis.data import taiwan_bankruptcy as tb
from fincrisis.evaluation import cross_validation as cv
from fincrisis.evaluation import metrics as mt
from fincrisis.models.distress import DIAGNOSTICS, FAMILIES, build_pipelines, large_value_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_CAVEAT = (
    "Label 1 is the minority class and is treated as the class to detect. "
    "The meanings of labels 0 and 1 are not documented by UCI and remain unverified; "
    "no result here should be read as predicting bankruptcy."
)
CV_FILES = ["cv_results.csv", "oof_predictions.csv", "permutation_importance.csv", "selection.json", "cv_report.md"]
FINAL_FILES = ["test_predictions.csv", "test_metrics.json", "final_report.md"]
THRESHOLD_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SMOKE_OVERRIDES = {"cv_folds": 2, "cv_repeats": 1, "bootstrap_resamples": 50, "permutation_rounds": 1}


# ---------- helpers ----------

def load_settings(smoke: bool) -> tuple[dict, int, dict]:
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    distress = dict(settings["company_distress"])
    if distress["threshold_rule"] != "max_f1":
        raise ValueError(f"Unsupported threshold_rule {distress['threshold_rule']!r}; only 'max_f1' is implemented")
    if smoke:
        distress.update(SMOKE_OVERRIDES)
        distress["xgboost"] = {**distress["xgboost"], "n_estimators": 20}
    return distress, settings["project"]["random_seed"], settings["paths"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_checksums(input_dir: Path, raw_dir: Path | None) -> dict:
    """Verify processed files (and the raw CSV when checking the project's own data)."""
    result = {"processed_sha256": tb.processed_checksums(input_dir), "raw_verified": False}
    if raw_dir is not None:
        recorded = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))["csv_sha256"]
        actual = sha256_file(raw_dir / "data.csv")
        if actual != recorded:
            raise tb.ValidationError(f"Raw CSV checksum mismatch: {actual} != {recorded}")
        result.update({"raw_sha256": actual, "raw_verified": True})
    return result


def git_state() -> dict:
    def run(*args):
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def write_outputs(outputs: dict[Path, str | bytes]) -> None:
    """Write all outputs to temporary files, then move them into place; clean up on error."""
    temps = {path: path.with_suffix(path.suffix + ".part") for path in outputs}
    try:
        for path, content in outputs.items():
            if isinstance(content, bytes):
                temps[path].write_bytes(content)
            else:
                temps[path].write_text(content, encoding="utf-8", newline="\n")
        for path, tmp in temps.items():
            tmp.replace(path)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)


def fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "None."
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def mean_sd(values) -> str:
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}" if len(values) > 1 else f"{values.mean():.4f}"


def json_text(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


# ---------- stage: cv ----------

def run_cv(args, distress: dict, seed: int, input_dir: Path, raw_dir: Path | None, out_dir: Path) -> int:
    out_paths = {name: out_dir / name for name in CV_FILES}
    existing = [str(p) for p in out_paths.values() if p.exists()]
    if existing and not args.force:
        print(f"CV output(s) already exist: {existing}\nNothing was written. Use --force to overwrite.")
        return 1

    checksums_before = input_checksums(input_dir, raw_dir)
    X, y, row_ids = tb.load_processed(input_dir, "train")  # training rows only
    large_cols = large_value_columns(X, distress["large_value_threshold"])  # training rows only
    # Milestone 3 listed large-value columns over all rows (train + test); recorded for comparison only.
    all_rows_meta = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    large_all_rows = all_rows_meta.get("large_value_features_unchanged")
    pipelines = build_pipelines(distress, large_cols, seed)
    n_splits = distress["cv_folds"]
    folds = cv.make_folds(y, n_splits, distress["cv_repeats"], seed)
    print(f"Training rows: {len(y)} (label 1: {int(y.sum())}); {len(folds)} folds x {len(pipelines)} configs")

    scores, oof, caught = cv.cross_validate_configs(pipelines, X, y, folds, n_splits)
    y_np = y.to_numpy()
    n_repeats = len(folds) // n_splits

    # Thresholds from pooled out-of-fold predictions (training data only).
    pooled_y = np.tile(y_np, n_repeats)
    thresholds, at_threshold = {}, {}
    for name, preds in oof.items():
        thresholds[name] = mt.select_threshold_max_f1(pooled_y, preds.ravel())
        at_threshold[name] = {
            label: [mt.threshold_metrics(y_np, preds[r], t) for r in range(n_repeats)]
            for label, t in (("max_f1", thresholds[name]), ("0.5", 0.5))
        }

    selection = {family: cv.select_treatment(scores, candidates) for family, candidates in FAMILIES.items()}
    selected = {family: s["selected"] for family, s in selection.items()}

    comparisons = [cv.paired_comparison(scores, a, b) for a, b in [
        ("lr_quantile", "lr_standard"), ("lr_large_as_missing", "lr_standard"),
        ("lr_quantile", "lr_large_as_missing"), ("xgb_large_as_missing", "xgb_raw"),
        ("lr_quantile_drop24", "lr_quantile"), ("xgb_raw_drop24", "xgb_raw"),
        (selected["logistic_regression"], "dummy"), (selected["xgboost"], "dummy"),
        (selected["xgboost"], selected["logistic_regression"]),
    ]]

    importance_rows = []
    first_repeat = folds[:n_splits]
    for family in ("logistic_regression", "xgboost"):
        name = selected[family]
        for r in cv.grouped_permutation_importance(pipelines[name], X, y, first_repeat, large_cols,
                                                   distress["permutation_rounds"], seed):
            importance_rows.append({"config": name, **r})
    importance = pd.DataFrame(importance_rows)

    checksums_after = input_checksums(input_dir, raw_dir)
    if checksums_after != checksums_before:
        print("Input checksums changed during the CV stage. Nothing was written.")
        return 1

    oof_frame = pd.concat([
        pd.DataFrame({"row_id": row_ids, "repeat": r, "config": name, "label": y_np, "proba": preds[r]})
        for name, preds in oof.items() for r in range(n_repeats)
    ])
    meta = {
        "stage": "cv", "smoke": args.smoke, "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_caveat": LABEL_CAVEAT, "positive_label": 1,
        "selection_rule": ("Per model family, the first treatment in preference order whose mean fold-by-fold "
                           "average-precision deficit to the best treatment is <= one standard error. "
                           f"Preference order: {FAMILIES}. Diagnostics {DIAGNOSTICS} are never selected."),
        "threshold_rule": "max F1 on pooled training out-of-fold predictions; 0.5 reported as reference",
        "selected": {family: {"config": selected[family], "threshold_max_f1": thresholds[selected[family]],
                              "evidence": selection[family]} for family in FAMILIES},
        "thresholds_max_f1": thresholds,
        "large_value_columns": large_cols,
        "large_value_columns_definition": (f"columns with any training-row value > {distress['large_value_threshold']}; "
                                           "derived from training rows only"),
        "large_value_columns_all_rows": large_all_rows,
        "large_only_outside_training": sorted(set(large_all_rows or []) - set(large_cols)),
        "settings": distress, "seed": seed,
        "training": {"rows": int(len(y)), "label_1": int(y.sum()), "folds": len(folds), "n_splits": n_splits,
                     "n_repeats": n_repeats},
        "input": checksums_before, "git": git_state(), "versions": versions(), "warnings": caught,
    }
    report = build_cv_report(meta, scores, at_threshold, comparisons, importance, oof, y_np, n_repeats)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs({
        out_paths["cv_results.csv"]: scores.to_csv(index=False, lineterminator="\n"),
        out_paths["oof_predictions.csv"]: oof_frame.to_csv(index=False, lineterminator="\n"),
        out_paths["permutation_importance.csv"]: importance.to_csv(index=False, lineterminator="\n"),
        out_paths["selection.json"]: json_text(meta),
        out_paths["cv_report.md"]: report,
    })
    print(f"Saved CV outputs to {out_dir}")
    print("Selected:", {f: f"{c} (threshold {thresholds[c]:.4f})" for f, c in selected.items()})
    print("Input checksums unchanged. The test set was not loaded.")
    return 0


def build_cv_report(meta, scores, at_threshold, comparisons, importance, oof, y_np, n_repeats) -> str:
    configs = list(oof)
    ranking = [[c] + [mean_sd(scores.loc[scores.config == c, m]) for m in
                      ("average_precision", "roc_auc", "brier", "log_loss",
                       "precision_at_0.5", "recall_at_0.5", "f1_at_0.5")] for c in configs]
    thresh_rows = []
    for c in configs:
        for label in ("max_f1", "0.5"):
            reps = at_threshold[c][label]
            thresh_rows.append([c, label, f"{reps[0]['threshold']:.4f}",
                                mean_sd([r["precision"] for r in reps]), mean_sd([r["recall"] for r in reps]),
                                mean_sd([r["f1"] for r in reps]),
                                " / ".join(f"{np.mean([r[k] for r in reps]):.1f}" for k in ("tn", "fp", "fn", "tp"))])
    comp_rows = [[c["a"], c["b"], f"{c['mean_difference']:+.4f}", f"{c['sd']:.4f}", f"{c['se']:.4f}",
                  f"{c['folds_a_better']}/{c['n_folds']}"] for c in comparisons]
    sel_rows = []
    for family, s in meta["selected"].items():
        ev = s["evidence"]
        means = ", ".join(f"{c} {m:.4f}" for c, m in ev["means"].items())
        checks = "; ".join(f"{c['a']} - {c['b']} = {c['mean_difference']:+.4f} (se {c['se']:.4f})"
                           for c in ev["comparisons"])
        evidence = f"{ev['reason']}. Mean AP: {means}" + (f". Tie-break checks: {checks}" if checks else "")
        sel_rows.append([family, s["config"], ev["best_by_mean"], f"{s['threshold_max_f1']:.4f}", evidence])
    imp_rows = [[c, mean_sd(g["base_average_precision"]), mean_sd(g["mean_drop"]),
                 mean_sd(g["gain_share"]) if g["gain_share"].notna().all() else "n/a"]
                for c, g in importance.groupby("config")]
    pooled_y = np.tile(y_np, n_repeats)
    detail = []
    for family in ("logistic_regression", "xgboost"):
        c = meta["selected"][family]["config"]
        proba = oof[c].ravel()
        grid = sorted(set(THRESHOLD_GRID) | {meta["thresholds_max_f1"][c]})
        detail += [f"### {c}: threshold analysis (pooled out-of-fold, all repeats)", "",
                   md_table(["Threshold", "Precision", "Recall", "F1", "False-positive rate", "Flagged"],
                            [[f"{r['threshold']:.4f}", r["precision"], r["recall"], r["f1"], r["false_positive_rate"],
                              r["tp"] + r["fp"]] for r in mt.threshold_table(pooled_y, proba, grid)]), "",
                   f"### {c}: calibration (pooled out-of-fold)", "",
                   md_table(["Predicted range", "Rows", "Mean predicted", "Observed share of label 1"],
                            [[r["bin"], r["n"], r["mean_predicted"], r["observed_rate"]]
                             for r in mt.calibration_table(pooled_y, proba)]), ""]
    warn_rows = [[c, "<br>".join(w)] for c, w in meta["warnings"].items() if w]
    t = meta["training"]
    lines = [
        "# Company Distress Baselines: Cross-Validation Report",
        "",
        f"> **Label caveat.** {LABEL_CAVEAT}",
        "",
        f"Generated (UTC): {meta['generated_at_utc']}" + ("  \n**SMOKE RUN (tiny settings, not for review)**" if meta["smoke"] else ""),
        "",
        "## Setup",
        "",
        f"- Training rows only: {t['rows']} (label 1: {t['label_1']}). The held-out test set was not loaded.",
        f"- Repeated stratified CV: {t['n_splits']} folds x {t['n_repeats']} repeats = {t['folds']} folds, seed {meta['seed']}.",
        "- Fixed model settings, no tuning. All transformations inside pipelines, refitted per training fold.",
        "- Class weights from each fit's labels (`class_weight='balanced'`; XGBoost `scale_pos_weight` = count(0)/count(1)).",
        f"- Large-value columns (any training-row value > {meta['settings']['large_value_threshold']}): "
        f"{len(meta['large_value_columns'])}. Their values are unexplained, not assumed erroneous."
        + (f" Over all rows (Milestone 3) there are {len(meta['large_value_columns_all_rows'])}; "
           f"{', '.join(meta['large_only_outside_training'])} has values above the threshold only in test rows, "
           "so the training-derived list excludes it." if meta["large_only_outside_training"] else ""),
        "- The `_drop24` diagnostics drop the training-derived large-value columns "
        f"({len(meta['large_value_columns'])} here); the name refers to the original inspection.",
        f"- Git commit: `{meta['git']['commit']}` (uncommitted changes: {meta['git']['dirty']}).",
        "",
        "## 1. Fold metrics (mean ± standard deviation across folds)",
        "",
        md_table(["Config", "Average precision", "ROC-AUC", "Brier", "Log-loss", "Precision@0.5", "Recall@0.5", "F1@0.5"], ranking),
        "",
        "## 2. Thresholds (out-of-fold predictions, mean ± sd across repeats)",
        "",
        "The max-F1 threshold is chosen on the same out-of-fold predictions it is evaluated on, so its "
        "precision/recall/F1 here are optimistic. Confusion counts are means per repeat (TN / FP / FN / TP).",
        "",
        md_table(["Config", "Rule", "Threshold", "Precision", "Recall", "F1", "TN / FP / FN / TP"], thresh_rows),
        "",
        "## 3. Paired comparisons (average precision, A - B, same folds)",
        "",
        "Repeated-CV folds overlap, so the standard error understates uncertainty. Drop-24 rows are "
        "experimental ablations: a change in score does not show that those features are defective or valid. "
        "Folds where A is better count strict improvements.",
        "",
        md_table(["A", "B", "Mean difference", "SD", "SE", "Folds A better"], comp_rows),
        "",
        "## 4. Selected configurations (frozen in selection.json)",
        "",
        f"Rule: {meta['selection_rule']}",
        "",
        md_table(["Family", "Selected", "Best by mean", "Max-F1 threshold", "Evidence"], sel_rows),
        "",
        "## 5. Influence of the large-value features (first repeat, training folds)",
        "",
        "Mean drop in average precision when all large-value columns are shuffled together in the validation "
        "fold; XGBoost gain share is the fraction of total split gain from those columns.",
        "",
        md_table(["Config", "Base average precision", "Drop when shuffled", "XGBoost gain share"], imp_rows),
        "",
        "## 6. Threshold analysis and calibration of selected configurations",
        "",
        *detail,
        "## 7. Warnings raised during fitting",
        "",
        md_table(["Config", "Warnings"], warn_rows),
        "",
        "## Limitations",
        "",
        f"- Only {t['label_1']} label-1 rows in training (about {t['label_1'] // t['n_splits']} per validation fold): "
        "fold metrics vary widely and small differences are not meaningful.",
        "- Repeated CV reuses the same rows; its fold scores are correlated and the reported spread is not a confidence interval.",
        "- No company identifier or date: rows from the same company may fall in different folds, which can inflate scores.",
        "- Feature units and definitions are undocumented; the large values remain unexplained.",
        "- These are cross-validation estimates on training data. The held-out test set has not been evaluated.",
        "",
    ]
    return "\n".join(lines)


# ---------- stage: final ----------

def run_final(args, input_dir: Path, raw_dir: Path | None, out_dir: Path, artifacts_dir: Path) -> int:
    selection_path = out_dir / "selection.json"
    if not selection_path.exists():
        print(f"{selection_path} not found. Run --stage cv and review it first. Nothing was written.")
        return 1
    out_paths = {name: out_dir / name for name in FINAL_FILES}
    existing = [str(p) for p in out_paths.values() if p.exists()]
    if existing and not args.force:
        print(f"Final output(s) already exist: {existing}\nNothing was written. Use --force to overwrite.")
        return 1

    meta_cv = json.loads(selection_path.read_text(encoding="utf-8"))
    checksums_before = input_checksums(input_dir, raw_dir)
    if checksums_before["processed_sha256"] != meta_cv["input"]["processed_sha256"]:
        print("Processed data differs from the data used in the CV stage. Nothing was written.")
        return 1

    distress, seed = meta_cv["settings"], meta_cv["seed"]
    X_train, y_train, _ = tb.load_processed(input_dir, "train")
    X_test, y_test, test_ids = tb.load_processed(input_dir, "test")
    pipelines = build_pipelines(distress, meta_cv["large_value_columns"], seed)
    n_boot = distress["bootstrap_resamples"]
    y_t = y_test.to_numpy()

    results, probas, models = {}, {}, {}
    for family, sel in meta_cv["selected"].items():
        name = sel["config"]
        model = pipelines[name].fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        models[name], probas[name] = model, proba
        results[family] = {
            "config": name, **mt.ranking_metrics(y_t, proba),
            "at_max_f1_threshold": mt.threshold_metrics(y_t, proba, sel["threshold_max_f1"]),
            "at_0.5": mt.threshold_metrics(y_t, proba, 0.5),
            "bootstrap_95ci_at_max_f1": mt.bootstrap_ci(y_t, proba, sel["threshold_max_f1"], n_boot, seed),
        }
    sel_names = {f: s["config"] for f, s in meta_cv["selected"].items()}
    differences = {
        f"{a} - {b}": mt.paired_bootstrap_diff(y_t, probas[a], probas[b], n_boot, seed)
        for a, b in [(sel_names["logistic_regression"], sel_names["dummy"]),
                     (sel_names["xgboost"], sel_names["dummy"]),
                     (sel_names["xgboost"], sel_names["logistic_regression"])]
    }

    if input_checksums(input_dir, raw_dir) != checksums_before:
        print("Input checksums changed during the final stage. Nothing was written.")
        return 1

    meta = {"stage": "final", "smoke": meta_cv["smoke"], "label_caveat": LABEL_CAVEAT,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "test_rows": int(len(y_t)), "test_label_1": int(y_t.sum()), "results": results,
            "paired_average_precision_differences": differences, "bootstrap_resamples": n_boot,
            "thresholds_frozen_from_cv": {f: s["threshold_max_f1"] for f, s in meta_cv["selected"].items()},
            "input": checksums_before, "git": git_state(), "versions": versions()}
    predictions = pd.DataFrame({"row_id": test_ids, "label": y_t, **{n: p for n, p in probas.items()}})

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for name, model in models.items():
        path = artifacts_dir / f"{name}.joblib"
        tmp = path.with_suffix(".joblib.part")
        try:
            joblib.dump(model, tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        model_files[name] = {"file": path.name, "sha256": sha256_file(path)}
    write_outputs({
        artifacts_dir / "model_metadata.json": json_text({
            "models": model_files, "trained_on": "full training split", "label_caveat": LABEL_CAVEAT,
            "note": "joblib files can execute code when loaded; load only files created by this project.",
            "settings": distress, "seed": seed, "git": meta["git"], "versions": meta["versions"]}),
        out_paths["test_predictions.csv"]: predictions.to_csv(index=False, lineterminator="\n"),
        out_paths["test_metrics.json"]: json_text(meta),
        out_paths["final_report.md"]: build_final_report(meta),
    })
    print(f"Saved final outputs to {out_dir} and models to {artifacts_dir}")
    print("Input checksums unchanged.")
    return 0


def build_final_report(meta: dict) -> str:
    rows = []
    for family, r in meta["results"].items():
        ci = r["bootstrap_95ci_at_max_f1"]
        m = r["at_max_f1_threshold"]
        rows.append([family, r["config"], f"{r['average_precision']:.4f} [{ci['average_precision'][0]:.3f}, {ci['average_precision'][1]:.3f}]",
                     f"{r['roc_auc']:.4f} [{ci['roc_auc'][0]:.3f}, {ci['roc_auc'][1]:.3f}]", r["brier"], r["log_loss"],
                     f"{m['threshold']:.4f}", f"{m['precision']:.3f} [{ci['precision'][0]:.3f}, {ci['precision'][1]:.3f}]",
                     f"{m['recall']:.3f} [{ci['recall'][0]:.3f}, {ci['recall'][1]:.3f}]",
                     f"{m['tn']} / {m['fp']} / {m['fn']} / {m['tp']}"])
    ref = [[f, r["config"], r["at_0.5"]["precision"], r["at_0.5"]["recall"], r["at_0.5"]["f1"],
            f"{r['at_0.5']['tn']} / {r['at_0.5']['fp']} / {r['at_0.5']['fn']} / {r['at_0.5']['tp']}"]
           for f, r in meta["results"].items()]
    diffs = [[k, f"{v['mean_difference']:+.4f}", f"[{v['interval'][0]:+.4f}, {v['interval'][1]:+.4f}]"]
             for k, v in meta["paired_average_precision_differences"].items()]
    return "\n".join([
        "# Company Distress Baselines: Held-out Test Evaluation",
        "",
        f"> **Label caveat.** {LABEL_CAVEAT}",
        "",
        f"Generated (UTC): {meta['generated_at_utc']}" + ("  \n**SMOKE RUN (tiny settings, not for review)**" if meta["smoke"] else ""),
        "",
        f"Test rows: {meta['test_rows']} (label 1: {meta['test_label_1']}). Treatments and thresholds were frozen "
        "in the CV stage; no test label was used to choose anything. All models are reported; no winner is chosen.",
        "",
        "## Results at the frozen max-F1 threshold (95% stratified bootstrap intervals)",
        "",
        md_table(["Family", "Config", "Average precision", "ROC-AUC", "Brier", "Log-loss", "Threshold",
                  "Precision", "Recall", "TN / FP / FN / TP"], rows),
        "",
        "## Reference: threshold 0.5",
        "",
        md_table(["Family", "Config", "Precision", "Recall", "F1", "TN / FP / FN / TP"], ref),
        "",
        "## Paired bootstrap differences in average precision",
        "",
        md_table(["Comparison", "Mean difference", "95% interval"], diffs),
        "",
        "## Limitations",
        "",
        f"- Only {meta['test_label_1']} label-1 rows in the test set; one row changes recall by "
        f"{100 / max(meta['test_label_1'], 1):.1f} percentage points. Intervals are wide and approximate.",
        "- No company identifier or date; results may be optimistic and do not reflect performance over time.",
        "- The meaning of the labels and the cause of the large feature values are unverified.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs of this stage")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="tiny settings for automated tests")
    args = parser.parse_args()

    distress, seed, paths = load_settings(args.smoke)
    default_input = PROJECT_ROOT / paths["interim_data"] / "taiwan_bankruptcy"
    input_dir = args.input_dir or default_input
    # The raw CSV is verified whenever the project's own processed data is used.
    raw_dir = PROJECT_ROOT / paths["raw_data"] / "taiwan_bankruptcy" if input_dir.resolve() == default_input.resolve() else None
    out_dir = args.output_dir or PROJECT_ROOT / paths["data_reports"] / "taiwan_bankruptcy" / "baselines"
    artifacts_dir = args.artifacts_dir or PROJECT_ROOT / paths["models"] / "taiwan_bankruptcy"
    try:
        if args.stage == "cv":
            return run_cv(args, distress, seed, input_dir, raw_dir, out_dir)
        return run_final(args, input_dir, raw_dir, out_dir, artifacts_dir)
    except tb.ValidationError as error:
        print(f"Validation failed: {error}\nNothing was written.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
