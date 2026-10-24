"""Market-stress models on the Phase B dataset (docs/decisions.md, D12 and D13).

Usage:
    python scripts/train_market_stress.py --stage cv       # development rows only
    python scripts/train_market_stress.py --stage final    # after review: one test evaluation
    Options: --force (overwrite this stage's outputs)

Stage "cv" passes only development rows (before the first test session minus
the purge) to the modelling code, runs the 15 walk-forward folds, and freezes
thresholds and the primary model in selection.json. It refuses to replace the
selection once the test set has been evaluated, or to change frozen values
without --accept-changed-selection. Stage "final" requires a complete
selection.json produced by the current, committed code, fits each model on
all development rows, and evaluates the 2015+ test set once.

Label 1 means the S&P 500 closes at least 5% below its running high within
the next 20 sessions (D2). Scores rank risk; they are not forecasts.
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

from fincrisis.data.market import MarketDataError, load_market_dataset
from fincrisis.evaluation import market_metrics as mm
from fincrisis.evaluation import metrics as mt
from fincrisis.evaluation.splits import walk_forward_splits
from fincrisis.features.market_features import FEATURES
from fincrisis.models import market_stress as ms_models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CV_FILES = ["oof_predictions.csv", "cv_metrics.json", "selection.json", "cv_report.md"]
FINAL_FILES = ["test_predictions.csv", "test_metrics.json", "final_report.md"]
CAVEAT = ("Scores rank the risk that the S&P 500 closes at least 5% below its running high within the next "
          "20 sessions (D2). They are research outputs, not forecasts or financial advice.")


def load_settings() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args):
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__}


def write_outputs(outputs: dict[Path, str]) -> None:
    """Write all outputs to temporary files, then move them into place; clean up on error."""
    temps = {path: path.with_suffix(path.suffix + ".part") for path in outputs}
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temps[path].write_text(content, encoding="utf-8", newline="\n")
        for path, tmp in temps.items():
            tmp.replace(path)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)


def json_text(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def fmt(value, digits=4) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:.{digits}f}"


def interval(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def make_splits(data: pd.DataFrame, settings: dict) -> dict:
    """Label-free: uses only dates and eligibility (D12)."""
    m = settings["market_stress"]
    years = list(range(m["validation_first_year"], m["validation_last_year"] + 1))
    return walk_forward_splits(data.index, data["eligible"].to_numpy(), years, m["test_start"],
                               settings["evaluation"]["purge_gap_trading_days"],
                               settings["market_stress_target"]["horizon_trading_days"])


def evaluate(y, scores: dict, kinds: dict, years, thresholds: dict, m: dict, seed: int, pairs) -> dict:
    """Pooled and per-year metrics, threshold metrics, calibration, and block-bootstrap intervals."""
    result = {}
    for name, score in scores.items():
        prob = ms_models.is_probability(kinds[name])
        per_year = mm.per_year_metrics(years, y, score, prob, m["min_positives_reliable"])
        result[name] = {
            "kind": kinds[name], **mm.score_metrics(y, score, prob),
            "mean_defined_yearly_average_precision": mm.mean_defined_average_precision(per_year),
            "per_year": per_year,
            "at_max_f1_threshold": mt.threshold_metrics(y, score, thresholds[name]),
            "at_0.5": mt.threshold_metrics(y, score, 0.5) if prob else None,
            "calibration": mt.calibration_table(y, score) if prob else None,
        }
    boot = {"primary": mm.block_bootstrap_ci(y, scores, m["bootstrap_block_length"], m["bootstrap_resamples"], seed,
                                             pairs=pairs)}
    for length in m["bootstrap_sensitivity_blocks"]:
        boot[f"block_{length}"] = mm.block_bootstrap_ci(y, scores, length, m["bootstrap_resamples"], seed)
    return {"models": result, "bootstrap": boot, "rows": int(len(y)), "label_1": int(np.sum(y)),
            "prevalence": float(np.mean(y))}


def comparison_pairs(names: list[str]) -> list[tuple[str, str]]:
    fitted = [n for n in ("logistic_regression", "xgboost") if n in names]
    others = [n for n in names if n not in fitted]
    pairs = [(f, o) for f in fitted for o in others]
    return pairs + ([("xgboost", "logistic_regression")] if len(fitted) == 2 else [])


PRIMARY_RULE = "highest pooled walk-forward average precision among non-dummy models (D13)"


def select_primary(results: dict) -> dict:
    """Primary model from validation results only; the dummy is a reference, not a candidate."""
    candidates = {n: r["average_precision"] for n, r in results["models"].items() if n != "dummy"}
    name = max(candidates, key=candidates.get)
    diffs = results["bootstrap"]["primary"]["differences"]
    evidence = {k: v for k, v in diffs.items() if name in k.split(" - ")}
    return {"name": name, "rule": PRIMARY_RULE,
            "ranking": sorted(([n, ap] for n, ap in candidates.items()), key=lambda x: -x[1]),
            "paired_differences_involving_primary": evidence}


def run_cv(dataset_dir, report_dir, settings, force, raw_check, accept_changed_selection=False) -> int:
    out = {name: report_dir / name for name in CV_FILES}
    tested = [str(report_dir / n) for n in FINAL_FILES if (report_dir / n).exists()]
    if tested:
        print(f"The held-out test set has already been evaluated with this selection: {tested}\n"
              "CV results and selection.json cannot be replaced after the test evaluation (not even with --force). "
              "Nothing was written.")
        return 1
    existing = [str(p) for p in out.values() if p.exists()]
    if existing and not force:
        print(f"CV output(s) already exist: {existing}\nNothing was written. Use --force to overwrite.")
        return 1
    previous = json.loads(out["selection.json"].read_text(encoding="utf-8")) if out["selection.json"].exists() else None
    raw_before = raw_check()
    data, meta = load_market_dataset(dataset_dir)
    checksum = meta["output"]["dataset.csv"]
    splits = make_splits(data, settings)
    dev_limit = splits["first_test_position"] - splits["purge"]
    dev = data.iloc[:dev_limit].copy()  # development rows only: no test-period row or label below this line
    del data
    if any(f["validation"].max() >= dev_limit or f["train"].max() >= dev_limit for f in splits["folds"]):
        raise MarketDataError("a fold reaches beyond the development rows")

    m, seed = settings["market_stress"], settings["project"]["random_seed"]
    models = ms_models.build_models(m, seed)
    kinds = {name: spec[0] for name, spec in models.items()}
    oof = ms_models.walk_forward_predict(models, dev, splits["folds"])
    y = oof["label"].to_numpy()
    thresholds = {name: mt.select_threshold_max_f1(y, oof[name].to_numpy()) for name in models}
    results = evaluate(y, {n: oof[n].to_numpy() for n in models}, kinds, oof["year"].to_numpy(), thresholds, m, seed,
                       comparison_pairs(list(models)))

    if load_market_dataset(dataset_dir)[1]["output"]["dataset.csv"] != checksum or raw_check() != raw_before:
        print("Input data changed during the CV stage. Nothing was written.")
        return 1
    primary = select_primary(results)
    if previous is not None and not accept_changed_selection:
        changed = []
        if previous.get("thresholds_max_f1") != thresholds:
            changed.append("thresholds_max_f1")
        if "primary_model" in previous and previous["primary_model"]["name"] != primary["name"]:
            changed.append("primary_model")
        if changed:
            print(f"Re-running CV would change the frozen {changed} in the existing selection.json. Nothing was "
                  "written. Pass --accept-changed-selection only if a new experiment is intended.")
            return 1
    selection = {
        "stage": "cv", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "caveat": CAVEAT, "features": FEATURES, "models": kinds, "thresholds_max_f1": thresholds,
        "primary_model": primary,
        "threshold_rule": "max F1 on pooled walk-forward validation predictions (development rows only)",
        "development": {"last_position": int(dev_limit - 1), "last_date": str(dev.index[-1].date()),
                        "validation_rows": int(len(oof)), "validation_label_1": int(y.sum())},
        "dataset_sha256": checksum, "raw_sha256": raw_before, "settings": settings["market_stress"],
        "horizon": settings["market_stress_target"]["horizon_trading_days"],
        "purge": settings["evaluation"]["purge_gap_trading_days"], "seed": seed,
        "git": git_state(), "versions": versions(),
    }
    oof_out = oof.copy()
    oof_out["date"] = pd.to_datetime(oof_out["date"]).dt.strftime("%Y-%m-%d")
    write_outputs({out["oof_predictions.csv"]: oof_out.to_csv(index=False, lineterminator="\n"),
                   out["cv_metrics.json"]: json_text(results), out["selection.json"]: json_text(selection),
                   out["cv_report.md"]: build_report("Walk-forward validation (development data, 2000-2014)",
                                                     results, thresholds, selection, final=False)})
    print(f"Saved CV outputs to {report_dir}. Validation rows {len(oof)}, label 1 {int(y.sum())}.")
    for name, r in results["models"].items():
        print(f"  {name:28} pooled AP {fmt(r['average_precision'])}  ROC-AUC {fmt(r['roc_auc'])}  threshold {thresholds[name]:.4f}")
    print(f"Primary model (by validation only): {primary['name']}")
    print("The test period was not passed to the modelling code.")
    return 0


def check_selection(selection: dict, models: dict, require_clean_code: bool, current_git: dict) -> list[str]:
    """Reasons the frozen selection cannot be used for the final stage (empty if it can)."""
    problems = []
    for key in ("models", "thresholds_max_f1", "features", "primary_model", "git", "dataset_sha256"):
        if key not in selection:
            problems.append(f"selection.json has no '{key}' (re-run --stage cv with the current code)")
    if problems:
        return problems
    if selection["features"] != FEATURES:
        problems.append("selection.json features differ from the current FEATURES")
    kinds = {name: spec[0] for name, spec in models.items()}
    if selection["models"] != kinds:
        problems.append(f"models rebuilt from the frozen settings {kinds} differ from selection.json {selection['models']}")
    if set(selection["thresholds_max_f1"]) != set(models):
        problems.append("frozen thresholds do not cover exactly the selected models")
    if selection["primary_model"]["name"] not in models:
        problems.append("primary model is not among the models")
    if require_clean_code:
        if selection["git"].get("dirty") is not False:
            problems.append("selection.json was produced from uncommitted code; commit, then re-run --stage cv")
        if current_git.get("dirty") is not False:
            problems.append("the working tree has uncommitted changes; the final stage must run on committed code")
        if current_git.get("commit") != selection["git"].get("commit"):
            problems.append(f"current commit {current_git.get('commit')} differs from the CV commit "
                            f"{selection['git'].get('commit')}; re-run --stage cv on this commit first")
    return problems


def run_final(dataset_dir, report_dir, artifacts_dir, force, raw_check, require_clean_code=False) -> int:
    selection_path = report_dir / "selection.json"
    if not selection_path.exists():
        print(f"{selection_path} not found. Run --stage cv and review it first. Nothing was written.")
        return 1
    out = {name: report_dir / name for name in FINAL_FILES}
    existing = [str(p) for p in out.values() if p.exists()]
    if existing and not force:
        print(f"Final output(s) already exist: {existing}\nNothing was written. Use --force to overwrite.")
        return 1
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes.decode("utf-8"))
    if "settings" not in selection or "seed" not in selection:
        print("selection.json is incomplete. Nothing was written.")
        return 1
    problems = check_selection(selection, ms_models.build_models(selection["settings"], selection["seed"]),
                               require_clean_code, git_state())
    if problems:
        print("The frozen selection cannot be used:\n  " + "\n  ".join(problems) + "\nNothing was written.")
        return 1
    raw_before = raw_check()
    data, meta = load_market_dataset(dataset_dir)
    if meta["output"]["dataset.csv"] != selection["dataset_sha256"]:
        print("dataset.csv differs from the one used in the CV stage. Nothing was written.")
        return 1
    settings = {"market_stress": selection["settings"], "evaluation": {"purge_gap_trading_days": selection["purge"]},
                "market_stress_target": {"horizon_trading_days": selection["horizon"]}}
    splits = make_splits(data, settings)
    m, seed = selection["settings"], selection["seed"]
    models = ms_models.build_models(m, seed)
    if splits["development"].max() + splits["horizon"] >= splits["test"].min():
        raise MarketDataError("a development label overlaps the test period")
    dev_rows, test_rows = data.iloc[splits["development"]], data.iloc[splits["test"]]
    X_dev, X_test = ms_models.feature_matrix(dev_rows), ms_models.feature_matrix(test_rows)
    scores, fitted = {}, {}
    for name, spec in models.items():
        scores[name], fitted[name] = ms_models.fit_predict(spec, X_dev, dev_rows["target"], X_test)
    y = test_rows["target"].to_numpy()
    kinds = {name: spec[0] for name, spec in models.items()}
    results = evaluate(y, scores, kinds, test_rows.index.year.to_numpy(), selection["thresholds_max_f1"], m, seed,
                       comparison_pairs(list(models)))
    if load_market_dataset(dataset_dir)[1]["output"]["dataset.csv"] != selection["dataset_sha256"] or raw_check() != raw_before:
        print("Input data changed during the final stage. Nothing was written.")
        return 1
    if selection_path.read_bytes() != selection_bytes:
        print("selection.json changed during the final stage. Nothing was written.")
        return 1

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for name, model in fitted.items():
        if model is None:
            continue
        path = artifacts_dir / f"{name}.joblib"
        tmp = path.with_suffix(".joblib.part")
        try:
            joblib.dump(model, tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
        model_files[name] = {"file": path.name, "sha256": sha256_file(path)}
    info = {"stage": "final", "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "trained_on": {"rows": int(len(dev_rows)), "first_date": str(dev_rows.index[0].date()),
                           "last_date": str(dev_rows.index[-1].date())},
            "test": {"rows": int(len(test_rows)), "first_date": str(test_rows.index[0].date()),
                     "last_date": str(test_rows.index[-1].date())},
            "dataset_sha256": selection["dataset_sha256"], "git": git_state(), "versions": versions(),
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "selection_git": selection["git"], "primary_model_from_cv": selection["primary_model"]["name"]}
    predictions = pd.DataFrame({"date": test_rows.index.strftime("%Y-%m-%d"), "position": test_rows["position"].to_numpy(),
                                "label": y, **scores})
    write_outputs({
        artifacts_dir / "model_metadata.json": json_text({**info, "models": model_files, "features": FEATURES,
                                                          "note": "joblib files can execute code when loaded; load only files created by this project."}),
        out["test_predictions.csv"]: predictions.to_csv(index=False, lineterminator="\n"),
        out["test_metrics.json"]: json_text({**results, **info, "thresholds_frozen_from_cv": selection["thresholds_max_f1"]}),
        out["final_report.md"]: build_report(f"Held-out test ({info['test']['first_date']} .. {info['test']['last_date']})",
                                             results, selection["thresholds_max_f1"], selection, final=True),
    })
    print(f"Saved final outputs to {report_dir} and models to {artifacts_dir}.")
    return 0


def build_report(title: str, results: dict, thresholds: dict, selection: dict, final: bool) -> str:
    models = results["models"]
    boot = results["bootstrap"]["primary"]
    main = [f"| `{n}` | {r['kind']} | {fmt(r['average_precision'])} {interval(boot['average_precision'][n])} | "
            f"{fmt(r['roc_auc'])} | {fmt(r['brier'])} | {fmt(r['log_loss'])} | {fmt(r['mean_defined_yearly_average_precision'])} |"
            for n, r in models.items()]
    thr = []
    for n, r in models.items():
        t = r["at_max_f1_threshold"]
        thr.append(f"| `{n}` | {thresholds[n]:.4f} | {t['precision']:.3f} | {t['recall']:.3f} | {t['f1']:.3f} | "
                   f"{t['tn']} / {t['fp']} / {t['fn']} / {t['tp']} |")
        if r["at_0.5"]:
            h = r["at_0.5"]
            thr.append(f"| `{n}` @0.5 | 0.5000 | {h['precision']:.3f} | {h['recall']:.3f} | {h['f1']:.3f} | "
                       f"{h['tn']} / {h['fp']} / {h['fn']} / {h['tp']} |")
    names = list(models)
    years = [r["year"] for r in models[names[0]]["per_year"]]
    yearly = []
    for i, year in enumerate(years):
        base = models[names[0]]["per_year"][i]
        cells = " | ".join(fmt(models[n]["per_year"][i]["average_precision"], 3) for n in names)
        yearly.append(f"| {year} | {base['rows']} | {base['label_1']} | {base['prevalence']:.3f} | {cells} | {base['status']} |")
    diffs = [f"| {k} | {fmt(v['mean'])} | {interval(v['interval'])} |" for k, v in boot["differences"].items()]
    sens = [f"| {key.replace('block_', '')} | " + " | ".join(interval(b["average_precision"][n]) for n in names) + " |"
            for key, b in results["bootstrap"].items() if key != "primary"]
    calib = []
    for n, r in models.items():
        if r["calibration"]:
            calib += [f"### `{n}`", "", "| Predicted range | Rows | Mean predicted | Observed share of label 1 |", "|---|---|---|---|",
                      *[f"| {c['bin']} | {c['n']} | {c['mean_predicted']:.3f} | {c['observed_rate']:.3f} |" for c in r["calibration"]], ""]
    lines = [
        f"# Market-Stress Models: {title}",
        "",
        f"> {CAVEAT}",
        "",
        f"Rows: {results['rows']}, label 1: {results['label_1']} (prevalence {results['prevalence']:.3f}; this is the "
        "average precision of an uninformative score). Inputs: the 16 features of D12. Fixed settings, no tuning.",
        "",
        (f"Primary model, chosen on validation results only ({selection['primary_model']['rule']}): "
         f"`{selection['primary_model']['name']}`. " if "primary_model" in selection else "")
        + "Thresholds were chosen on the pooled walk-forward validation predictions and frozen in `selection.json`"
        + (". The test set was evaluated once at those thresholds; every model is reported, and test results "
           "do not change the primary model." if final else
           "; on the validation data they are optimistic because they were chosen there."),
        "",
        f"## Pooled metrics (95% block-bootstrap interval, {boot['block_length']}-session blocks, "
        f"{boot['resamples']} resamples, {boot['skipped_one_class']} skipped)",
        "",
        "| Model | Kind | Average precision [95% CI] | ROC-AUC | Brier | Log-loss | Mean of defined yearly AP |",
        "|---|---|---|---|---|---|---|",
        *main,
        "",
        "Rules output raw feature values, not probabilities, so Brier and log-loss do not apply to them.",
        "",
        "## Paired differences in average precision (same resamples)",
        "",
        "| Comparison | Mean difference | 95% interval |",
        "|---|---|---|",
        *diffs,
        "",
        "## Block-length sensitivity (95% intervals for average precision)",
        "",
        "| Block length | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
        *sens,
        "",
        "## Thresholds and confusion matrices (TN / FP / FN / TP)",
        "",
        "| Model | Threshold | Precision | Recall | F1 | TN / FP / FN / TP |",
        "|---|---|---|---|---|---|",
        *thr,
        "",
        "## Per-year average precision",
        "",
        "Undefined values (one class only) are `n/a`, not 0, and are excluded from the yearly mean. Years with few "
        "label-1 rows are flagged. Yearly scores are descriptive and were not used to choose anything.",
        "",
        "| Year | Rows | Label 1 | Prevalence | " + " | ".join(f"`{n}`" for n in names) + " | Status |",
        "|---|---|---|---|" + "---|" * len(names) + "---|",
        *yearly,
        "",
        "## Calibration (probability models)",
        "",
        *calib,
        "## Limitations",
        "",
        "- Label-1 rows overlap in 20-session windows and cluster in a small number of episodes; the effective sample "
        "is far smaller than the row count, and intervals are approximate.",
        "- Pooled validation scores mix 15 models trained on different windows.",
        "- Yahoo Finance data is unofficial. Results describe 1990-2026 history and need not hold in the future.",
        "",
    ]
    return "\n".join(lines)


def raw_checker(raw_dir: Path | None):
    def check():
        if raw_dir is None:
            return None
        meta = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
        result = {}
        for s in meta["series"].values():
            actual = sha256_file(raw_dir / s["file"])
            if actual != s["sha256"]:
                raise MarketDataError(f"raw snapshot {s['file']} checksum mismatch")
            result[s["file"]] = actual
        return result
    return check


def run(stage: str, dataset_dir: Path, report_dir: Path, artifacts_dir: Path, settings: dict, force: bool,
        raw_dir: Path | None = None, require_clean_code: bool = False, accept_changed_selection: bool = False) -> int:
    try:
        if settings["market_stress"]["threshold_rule"] != "max_f1":
            raise MarketDataError("only threshold_rule 'max_f1' is implemented")
        if stage == "cv":
            return run_cv(dataset_dir, report_dir, settings, force, raw_checker(raw_dir), accept_changed_selection)
        return run_final(dataset_dir, report_dir, artifacts_dir, force, raw_checker(raw_dir), require_clean_code)
    except (MarketDataError, ValueError) as error:
        print(f"Failed: {error}\nNothing was written.")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["cv", "final"], required=True)
    parser.add_argument("--force", action="store_true", help="overwrite this stage's existing outputs")
    parser.add_argument("--accept-changed-selection", action="store_true",
                        help="cv only: allow a re-run to change the frozen thresholds or primary model")
    args = parser.parse_args()
    settings = load_settings()
    paths = settings["paths"]
    return run(args.stage, PROJECT_ROOT / paths["processed_data"] / "market_stress",
               PROJECT_ROOT / paths["data_reports"] / "market_stress" / "models",
               PROJECT_ROOT / paths["models"] / "market_stress", settings, args.force,
               raw_dir=PROJECT_ROOT / paths["raw_data"] / "market", require_clean_code=True,
               accept_changed_selection=args.accept_changed_selection)


if __name__ == "__main__":
    sys.exit(main())
