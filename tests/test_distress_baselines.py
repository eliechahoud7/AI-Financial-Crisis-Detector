"""Tests for the company distress baselines (docs/decisions.md, D9).

Unit tests and the full two-stage integration test use synthetic processed data,
so no test ever evaluates the real held-out test set. One integration test runs
the CV stage on the real training rows (skipped if the data is absent).
"""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fincrisis.data import taiwan_bankruptcy as tb
from fincrisis.evaluation import cross_validation as cv
from fincrisis.evaluation import metrics as mt
from fincrisis.features.large_values import LargeValuesToMissing
from fincrisis.models.distress import BalancedXGBClassifier, build_pipelines, large_value_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "train_distress_baselines.py"
INTERIM = PROJECT_ROOT / "data" / "interim" / "taiwan_bankruptcy"
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "taiwan_bankruptcy" / "data.csv"
SETTINGS = {
    "large_value_threshold": 1.0,
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "xgboost": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "subsample": 1.0,
                "colsample_bytree": 1.0, "tree_method": "hist", "n_jobs": 1},
}
SEEN_ROWS: list[set] = []  # filled by RowRecorder; module-level so clones share it


class RowRecorder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        SEEN_ROWS.append(set(X.index))
        return self

    def transform(self, X):
        return X


def synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.12).astype("int64")
    signal = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    large = np.where(rng.random(n) < 0.3, rng.integers(1, 10, n) * 1e9, rng.random(n))
    return pd.DataFrame({
        "row_id": np.arange(n), "bankrupt": y, "signal": signal, "noise_a": rng.random(n),
        "noise_b": rng.random(n), "large_feature": large, "flag": (rng.random(n) < 0.02).astype("int64"),
    })


def write_processed_dir(directory: Path, frame: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    texts = {"data.csv": tb.to_csv_text(frame),
             "split.csv": tb.to_csv_text(tb.make_split(frame, test_size=0.2, seed=42))}
    for name, text in texts.items():
        (directory / name).write_text(text, encoding="utf-8", newline="\n")
    sha = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in texts.items()}
    (directory / "metadata.json").write_text(json.dumps({"output": {"sha256": sha}}), encoding="utf-8")


def dir_hashes(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


def run_script(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(SCRIPT), *args], capture_output=True, text=True)


class TestLargeValuesToMissing(unittest.TestCase):
    def test_replaces_only_values_above_threshold_and_keeps_input(self):
        X = pd.DataFrame({"a": [0.0, 0.5, 1.0, 1.5], "b": [6e9, 0.2, 2.0, 1.0]})
        before = X.copy()
        out = LargeValuesToMissing(1.0).fit_transform(X)
        expected = np.array([[0.0, np.nan], [0.5, 0.2], [1.0, np.nan], [np.nan, 1.0]])
        np.testing.assert_array_equal(out, expected)
        pd.testing.assert_frame_equal(X, before)

    def test_learns_nothing_from_fit_data(self):
        X = np.array([[0.5, 3.0]])
        a = LargeValuesToMissing().fit(np.zeros((5, 2))).transform(X)
        b = LargeValuesToMissing().fit(np.full((5, 2), 9e9)).transform(X)
        np.testing.assert_array_equal(a, b)


class TestBalancedXGB(unittest.TestCase):
    def test_weight_from_each_fit_labels(self):
        X = np.random.default_rng(0).random((40, 3))
        model = BalancedXGBClassifier(n_estimators=2)
        model.fit(X, np.array([1] * 4 + [0] * 36))
        self.assertAlmostEqual(model.get_params()["scale_pos_weight"], 9.0)
        model.fit(X, np.array([1] * 10 + [0] * 30))
        self.assertAlmostEqual(model.get_params()["scale_pos_weight"], 3.0)

    def test_clone_keeps_settings(self):
        copy = clone(BalancedXGBClassifier(n_estimators=7, max_depth=3))
        self.assertIsInstance(copy, BalancedXGBClassifier)
        self.assertEqual((copy.get_params()["n_estimators"], copy.get_params()["max_depth"]), (7, 3))


class TestPipelines(unittest.TestCase):
    def test_all_configs_fit_and_predict(self):
        frame = synthetic_frame()
        X, y = frame.drop(columns=["row_id", "bankrupt"]), frame["bankrupt"]
        large = large_value_columns(X, 1.0)
        self.assertEqual(large, ["large_feature"])
        pipelines = build_pipelines(SETTINGS, large, seed=42)
        self.assertEqual(len(pipelines), 8)
        for name, pipeline in pipelines.items():
            proba = clone(pipeline).fit(X, y).predict_proba(X)[:, 1]
            self.assertEqual(proba.shape, (len(X),), name)
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)), name)
        self.assertEqual(pipelines["lr_quantile"][-1].class_weight, "balanced")
        fitted = clone(pipelines["xgb_raw_drop24"]).fit(X, y)
        self.assertEqual(fitted[-1].n_features_in_, X.shape[1] - 1)


class TestMetrics(unittest.TestCase):
    def test_threshold_metrics_and_max_f1(self):
        y, p = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
        self.assertEqual(mt.select_threshold_max_f1(y, p), 0.35)
        m = mt.threshold_metrics(y, p, 0.35)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["f1"], 0.8)
        self.assertEqual(mt.threshold_metrics(y, p, 0.9)["precision"], 0.0)

    def test_calibration_table(self):
        rows = mt.calibration_table([0, 1, 0, 1], [0.05, 0.05, 0.95, 0.95])
        self.assertEqual([(r["n"], r["observed_rate"]) for r in rows], [(2, 0.5), (2, 0.5)])

    def test_bootstrap_deterministic_and_keeps_both_classes(self):
        y = np.array([0] * 30 + [1] * 3)
        p = np.linspace(0, 1, 33)
        for idx in mt.stratified_bootstrap_indices(y, 20, seed=1):
            self.assertEqual(int(y[idx].sum()), 3)
        self.assertEqual(mt.bootstrap_ci(y, p, 0.5, 50, seed=1), mt.bootstrap_ci(y, p, 0.5, 50, seed=1))
        diff = mt.paired_bootstrap_diff(y, p, p, 20, seed=1)
        self.assertEqual(diff["mean_difference"], 0.0)


class TestSelectionRule(unittest.TestCase):
    def scores(self, a, b):
        rows = [{"config": c, "repeat": 0, "fold": i, "average_precision": v}
                for c, values in (("simple", a), ("complex", b)) for i, v in enumerate(values)]
        return pd.DataFrame(rows)

    def test_prefers_simpler_when_within_one_standard_error(self):
        s = self.scores([0.50, 0.40, 0.60, 0.45], [0.55, 0.36, 0.63, 0.43])  # complex ahead by 0.005, se ~0.021
        result = cv.select_treatment(s, ["simple", "complex"])
        self.assertEqual((result["selected"], result["best_by_mean"]), ("simple", "complex"))

    def test_chooses_better_when_clearly_ahead(self):
        s = self.scores([0.40, 0.41, 0.42, 0.40], [0.50, 0.52, 0.51, 0.50])
        result = cv.select_treatment(s, ["simple", "complex"])
        self.assertEqual((result["selected"], result["best_by_mean"]), ("complex", "complex"))
        self.assertIn("more than one standard error behind", result["reason"])

    def test_reason_when_first_candidate_is_best(self):
        s = self.scores([0.50, 0.52, 0.51, 0.50], [0.40, 0.41, 0.42, 0.40])
        result = cv.select_treatment(s, ["simple", "complex"])
        self.assertEqual(result["selected"], "simple")
        self.assertEqual(result["reason"], "highest mean and first in preference order; no tie-break needed")
        self.assertEqual(set(result["means"]), {"simple", "complex"})
        self.assertEqual(cv.select_treatment(s, ["simple"])["reason"], "only candidate in this family")


class TestNoLeakage(unittest.TestCase):
    def test_cv_fits_see_only_their_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_processed_dir(Path(tmp), synthetic_frame())
            X, y, _ = tb.load_processed(Path(tmp), "train")
            _, _, test_ids = tb.load_processed(Path(tmp), "test")
        folds = cv.make_folds(y, 3, 2, seed=42)
        SEEN_ROWS.clear()
        pipeline = Pipeline([("record", RowRecorder()), ("model", DummyClassifier(strategy="prior"))])
        cv.cross_validate_configs({"recorded": pipeline}, X, y, folds, n_splits=3)
        self.assertEqual(len(SEEN_ROWS), len(folds))
        for seen, (train_idx, valid_idx) in zip(SEEN_ROWS, folds):
            self.assertEqual(seen, set(X.index[train_idx]))
            self.assertFalse(seen & set(X.index[valid_idx]))
            self.assertFalse(seen & set(test_ids))


class TestLoader(unittest.TestCase):
    def test_subsets_and_checksum_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = synthetic_frame()
            write_processed_dir(directory, frame)
            split = pd.read_csv(directory / "split.csv")
            X_train, y_train, train_ids = tb.load_processed(directory, "train")
            _, _, test_ids = tb.load_processed(directory, "test")
            self.assertEqual(set(train_ids), set(split.loc[split["split"] == "train", "row_id"]))
            self.assertFalse(set(train_ids) & set(test_ids))
            self.assertNotIn("bankrupt", X_train.columns)
            self.assertTrue(np.array_equal(y_train.to_numpy(), frame.loc[train_ids, "bankrupt"].to_numpy()))
            with (directory / "data.csv").open("a", encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaises(tb.ValidationError):
                tb.load_processed(directory, "train")


class TestTwoStageWorkflowSynthetic(unittest.TestCase):
    def test_cv_then_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data_dir, out, models = tmp / "processed", tmp / "reports", tmp / "models"
            write_processed_dir(data_dir, synthetic_frame())
            before = dir_hashes(data_dir)
            common = ["--input-dir", str(data_dir), "--output-dir", str(out), "--artifacts-dir", str(models), "--smoke"]

            refused = run_script("--stage", "final", *common)
            self.assertEqual(refused.returncode, 1)
            self.assertIn("selection.json", refused.stdout)

            first = run_script("--stage", "cv", *common)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            for name in ["cv_results.csv", "oof_predictions.csv", "permutation_importance.csv", "selection.json", "cv_report.md"]:
                self.assertTrue((out / name).exists(), name)
            self.assertIn("unverified", (out / "cv_report.md").read_text(encoding="utf-8"))
            selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
            self.assertIn("unverified", selection["label_caveat"])
            self.assertEqual(set(selection["selected"]), {"dummy", "logistic_regression", "xgboost"})
            self.assertNotIn(selection["selected"]["xgboost"]["config"], ["lr_quantile_drop24", "xgb_raw_drop24"])
            oof = pd.read_csv(out / "oof_predictions.csv")
            train_ids = set(pd.read_csv(data_dir / "split.csv").query("split == 'train'")["row_id"])
            self.assertEqual(set(oof["row_id"]), train_ids)

            self.assertEqual(run_script("--stage", "cv", *common).returncode, 1)  # no overwrite

            final = run_script("--stage", "final", *common)
            self.assertEqual(final.returncode, 0, final.stdout + final.stderr)
            metrics = json.loads((out / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metrics["results"]), {"dummy", "logistic_regression", "xgboost"})
            for family, s in selection["selected"].items():
                self.assertEqual(metrics["results"][family]["at_max_f1_threshold"]["threshold"], s["threshold_max_f1"])
            self.assertTrue((models / "model_metadata.json").exists())
            self.assertIn("unverified", (out / "final_report.md").read_text(encoding="utf-8"))
            self.assertEqual(run_script("--stage", "final", *common).returncode, 1)  # no overwrite
            self.assertEqual(dir_hashes(data_dir), before)


@unittest.skipUnless((INTERIM / "data.csv").exists() and RAW_CSV.exists(), "processed dataset not available")
class TestCvStageRealTrainingData(unittest.TestCase):
    def test_smoke_cv_uses_training_rows_only(self):
        raw_before, interim_before = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(), dir_hashes(INTERIM)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reports"
            result = run_script("--stage", "cv", "--smoke", "--output-dir", str(out), "--artifacts-dir", str(Path(tmp) / "m"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            oof = pd.read_csv(out / "oof_predictions.csv")
            split = pd.read_csv(INTERIM / "split.csv")
            self.assertEqual(set(oof["row_id"]), set(split.loc[split["split"] == "train", "row_id"]))
            self.assertEqual(oof["row_id"].nunique(), 5455)
            selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
            self.assertTrue(selection["input"]["raw_verified"])
            # Derived from training rows only: current_ratio's single large value is in the test rows.
            all_data = json.loads((INTERIM / "metadata.json").read_text(encoding="utf-8"))["large_value_features_unchanged"]
            self.assertLessEqual(set(selection["large_value_columns"]), set(all_data))
            self.assertEqual(len(selection["large_value_columns"]), 23)
            self.assertEqual(selection["large_only_outside_training"], ["current_ratio"])
            self.assertEqual(len(selection["large_value_columns_all_rows"]), 24)
            self.assertFalse(Path(tmp, "m").exists())  # CV stage writes no models
        self.assertEqual(hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(), raw_before)
        self.assertEqual(dir_hashes(INTERIM), interim_before)


if __name__ == "__main__":
    unittest.main()
