"""Tests for market-stress models and evaluation (docs/decisions.md, D13).

Unit tests and the full two-stage workflow use synthetic data, so no test
evaluates the real held-out test period. One test runs the CV stage on the
real development rows (skipped if the dataset is absent).
"""

import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from fincrisis.data.market import MarketDataError, load_market_dataset
from fincrisis.evaluation import market_metrics as mm
from fincrisis.evaluation.splits import walk_forward_splits
from fincrisis.features.market_features import FEATURES
from fincrisis.models import market_stress as msm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_market_dataset import SETTINGS as BUILD_SETTINGS, write_snapshot  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tms, bmd = load_script("train_market_stress"), load_script("build_market_dataset")
MODEL_SETTINGS = {
    "rule_baselines": ["vix_pct_252", "downside_vol_20"], "threshold_rule": "max_f1", "min_positives_reliable": 10,
    "bootstrap_resamples": 30, "bootstrap_block_length": 20, "bootstrap_sensitivity_blocks": [10],
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "xgboost": {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2, "subsample": 1.0, "colsample_bytree": 1.0,
                "min_child_weight": 1, "tree_method": "hist", "n_jobs": 1},
}
SEEN_ROWS: list[set] = []


class RowRecorder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        SEEN_ROWS.append(set(X.index))
        return self

    def transform(self, X):
        return X


def synthetic_frame(n_years=(1995, 2005), warm_up=30, seed=0) -> pd.DataFrame:
    dates = pd.bdate_range(f"{n_years[0]}-01-02", f"{n_years[1]}-12-30")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(len(dates), len(FEATURES))), index=dates, columns=FEATURES)
    frame["vix_pct_252"] = rng.random(len(dates))
    frame["downside_vol_20"] = rng.random(len(dates))
    frame.iloc[:warm_up, :] = np.nan
    frame.insert(0, "position", np.arange(len(dates)))
    frame["target"] = (rng.random(len(dates)) < 0.25).astype(int)
    frame["eligible"] = ~frame[FEATURES].isna().any(axis=1)
    frame["missing_features"] = ""
    return frame


def splits_for(frame, years=range(1998, 2003), test_start="2003-01-01"):
    return walk_forward_splits(frame.index, frame["eligible"].to_numpy(), list(years), test_start, 20, 20)


def full_settings(model_settings=MODEL_SETTINGS) -> dict:
    s = copy.deepcopy(BUILD_SETTINGS)
    s["market_stress"].update(copy.deepcopy(model_settings))
    s["project"] = {"random_seed": 42}
    return s


class TestFeatureMatrix(unittest.TestCase):
    def test_exactly_the_sixteen_features(self):
        frame = synthetic_frame().iloc[100:]
        X = msm.feature_matrix(frame)
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(len(FEATURES), 16)
        for column in msm.BOOKKEEPING + ["date"]:
            self.assertNotIn(column, X.columns)

    def test_rejects_missing_values_and_columns(self):
        frame = synthetic_frame()
        with self.assertRaises(ValueError):
            msm.feature_matrix(frame)  # warm-up rows contain NaN
        with self.assertRaises(ValueError):
            msm.feature_matrix(frame.iloc[100:].drop(columns="ret_1"))

    def test_models(self):
        models = msm.build_models(MODEL_SETTINGS, 42)
        self.assertEqual(list(models), ["dummy", "rule_vix_pct_252", "rule_downside_vol_20", "logistic_regression", "xgboost"])
        bad = copy.deepcopy(MODEL_SETTINGS)
        bad["rule_baselines"] = ["target"]
        with self.assertRaises(ValueError):
            msm.build_models(bad, 42)


class TestWalkForward(unittest.TestCase):
    def test_each_fit_sees_only_its_training_rows(self):
        frame = synthetic_frame()
        splits = splits_for(frame)
        dev = frame.iloc[:splits["first_test_position"] - 20]
        SEEN_ROWS.clear()
        recorder = ("fit", Pipeline([("record", RowRecorder()), ("model", DummyClassifier(strategy="prior"))]))
        oof = msm.walk_forward_predict({"recorded": recorder}, dev, splits["folds"])
        self.assertEqual(len(SEEN_ROWS), len(splits["folds"]))
        for seen, fold in zip(SEEN_ROWS, splits["folds"]):
            self.assertEqual(seen, set(frame.index[fold["train"]]))
            self.assertFalse(seen & set(frame.index[fold["validation"]]))
            self.assertLess(max(frame.index.get_indexer(list(seen))) + 20, fold["validation"].min())
        self.assertEqual(sorted(oof["position"]), sorted(np.concatenate([f["validation"] for f in splits["folds"]])))

    def test_rule_scores_are_feature_values_and_rows_predicted_once(self):
        frame = synthetic_frame()
        splits = splits_for(frame)
        dev = frame.iloc[:splits["first_test_position"] - 20]
        oof = msm.walk_forward_predict(msm.build_models(MODEL_SETTINGS, 42), dev, splits["folds"])
        self.assertTrue(oof["position"].is_unique)
        np.testing.assert_array_equal(oof["rule_vix_pct_252"], frame["vix_pct_252"].to_numpy()[oof["position"]])
        np.testing.assert_array_equal(oof["label"], frame["target"].to_numpy()[oof["position"]])
        self.assertTrue(((oof["dummy"] > 0) & (oof["dummy"] < 1)).all())

    def test_requires_frame_starting_at_position_zero(self):
        frame = synthetic_frame()
        with self.assertRaises(ValueError):
            msm.walk_forward_predict({}, frame.iloc[5:], splits_for(frame)["folds"])


class TestMetrics(unittest.TestCase):
    def test_single_class_metrics_are_undefined_not_zero(self):
        m = mm.score_metrics(np.zeros(10), np.linspace(0, 1, 10), True)
        self.assertEqual(m, {"average_precision": None, "roc_auc": None, "brier": None, "log_loss": None})

    def test_per_year_status_and_defined_mean(self):
        years = np.repeat([2004, 2005, 2013], 50)
        y = np.zeros(150)
        y[:20] = 1           # 2004: 20 positives
        y[100:102] = 1       # 2013: 2 positives; 2005: none
        score = np.linspace(0, 1, 150)
        rows = mm.per_year_metrics(years, y, score, False, 10)
        status = {r["year"]: r["status"] for r in rows}
        self.assertEqual(status[2004], "ok")
        self.assertTrue(status[2005].startswith("undefined"))
        self.assertTrue(status[2013].startswith("unreliable"))
        self.assertIsNone(rows[1]["average_precision"])
        defined = [r["average_precision"] for r in rows if r["average_precision"] is not None]
        self.assertAlmostEqual(mm.mean_defined_average_precision(rows), np.mean(defined))

    def test_block_indices_are_wrapping_contiguous_blocks(self):
        first = list(mm.circular_block_indices(25, 10, 3, seed=1))
        again = list(mm.circular_block_indices(25, 10, 3, seed=1))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, again)))
        for idx in first:
            self.assertEqual(len(idx), 25)
            for start in range(0, 25, 10):
                block = idx[start:start + 10]
                np.testing.assert_array_equal(np.diff(block) % 25, np.ones(len(block) - 1))
        with self.assertRaises(ValueError):
            next(mm.circular_block_indices(5, 10, 1, seed=0))

    def test_block_bootstrap_paired_difference_of_identical_scores_is_zero(self):
        rng = np.random.default_rng(0)
        y = (rng.random(300) < 0.3).astype(int)
        s = rng.random(300)
        result = mm.block_bootstrap_ci(y, {"a": s, "b": s}, 30, 50, seed=0, pairs=[("a", "b")])
        self.assertEqual(result["differences"]["a - b"]["interval"], [0.0, 0.0])
        self.assertEqual(result["average_precision"]["a"], result["average_precision"]["b"])

    def test_block_bootstrap_skips_one_class_resamples(self):
        y = np.zeros(100, dtype=int)
        y[:3] = 1
        result = mm.block_bootstrap_ci(y, {"a": np.linspace(0, 1, 100)}, 5, 50, seed=0)
        self.assertGreater(result["skipped_one_class"], 0)


class TestLoader(unittest.TestCase):
    def test_checksum_mismatch_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(bmd.run(tmp / "raw", tmp / "processed", tmp / "reports", BUILD_SETTINGS, False), 0)
            frame, _ = load_market_dataset(tmp / "processed")
            self.assertEqual(frame["eligible"].dtype, bool)
            with (tmp / "processed" / "dataset.csv").open("a", encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaises(MarketDataError):
                load_market_dataset(tmp / "processed")


class TestTwoStageWorkflowSynthetic(unittest.TestCase):
    def test_cv_then_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            settings = full_settings()
            quiet = io.StringIO()
            with redirect_stdout(quiet):
                self.assertEqual(bmd.run(tmp / "raw", tmp / "processed", tmp / "reports", settings, False), 0)
            data_hash = hashlib.sha256((tmp / "processed" / "dataset.csv").read_bytes()).hexdigest()
            reports, models = tmp / "reports" / "models", tmp / "artifacts"
            call = lambda stage, force=False: tms.run(stage, tmp / "processed", reports, models, settings, force, raw_dir=tmp / "raw")
            with redirect_stdout(quiet):
                self.assertEqual(call("final"), 1)                     # no selection yet
                self.assertEqual(call("cv"), 0)
                self.assertEqual(call("cv"), 1)                        # no overwrite
            selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
            oof = pd.read_csv(reports / "oof_predictions.csv")
            frame, _ = load_market_dataset(tmp / "processed")
            t0 = int(np.flatnonzero(frame.index >= "2002-01-01")[0])
            self.assertLess(oof["position"].max(), t0 - 20)            # nothing from the test period
            self.assertEqual(selection["features"], FEATURES)
            cv_metrics = json.loads((reports / "cv_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(set(cv_metrics["models"]), set(selection["models"]))
            self.assertIsNone(cv_metrics["models"]["rule_vix_pct_252"]["brier"])
            self.assertFalse(models.exists())                          # CV stage saves no models
            with redirect_stdout(quiet):
                self.assertEqual(call("final"), 0)
                self.assertEqual(call("final"), 1)                     # no overwrite
            test_metrics = json.loads((reports / "test_metrics.json").read_text(encoding="utf-8"))
            for name, threshold in selection["thresholds_max_f1"].items():
                self.assertEqual(test_metrics["models"][name]["at_max_f1_threshold"]["threshold"], threshold)
            self.assertTrue((models / "xgboost.joblib").exists() and (models / "model_metadata.json").exists())
            self.assertEqual(hashlib.sha256((tmp / "processed" / "dataset.csv").read_bytes()).hexdigest(), data_hash)
            self.assertEqual(test_metrics["primary_model_from_cv"], selection["primary_model"]["name"])
            self.assertEqual(test_metrics["selection_sha256"],
                             hashlib.sha256((reports / "selection.json").read_bytes()).hexdigest())
            selection_bytes = (reports / "selection.json").read_bytes()
            with redirect_stdout(quiet):
                self.assertEqual(call("cv", force=True), 1)            # selection is locked after the test run
            self.assertEqual((reports / "selection.json").read_bytes(), selection_bytes)


def built_workspace(tmp: Path):
    write_snapshot(tmp / "raw")
    settings = full_settings()
    with redirect_stdout(io.StringIO()):
        assert bmd.run(tmp / "raw", tmp / "processed", tmp / "reports", settings, False) == 0
    reports, models = tmp / "reports" / "models", tmp / "artifacts"

    def call(stage, force=False, **kwargs):
        with redirect_stdout(io.StringIO()):
            return tms.run(stage, tmp / "processed", reports, models, settings, force, raw_dir=tmp / "raw", **kwargs)
    return reports, call


class TestSelectionSafeguards(unittest.TestCase):
    def test_primary_model_is_best_non_dummy_validation_score(self):
        results = {"models": {"dummy": {"average_precision": 0.9}, "a": {"average_precision": 0.4},
                              "b": {"average_precision": 0.6}},
                   "bootstrap": {"primary": {"differences": {"b - a": {"mean": 0.2}, "a - dummy": {"mean": 0.1}}}}}
        primary = tms.select_primary(results)
        self.assertEqual(primary["name"], "b")
        self.assertEqual(primary["ranking"], [["b", 0.6], ["a", 0.4]])
        self.assertEqual(set(primary["paired_differences_involving_primary"]), {"b - a"})

    def test_rerun_cannot_silently_change_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports, call = built_workspace(Path(tmp))
            self.assertEqual(call("cv"), 0)
            selection = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
            self.assertEqual(call("cv", force=True), 0)               # identical re-run is allowed
            selection["thresholds_max_f1"]["xgboost"] += 0.01         # pretend the frozen value was different
            (reports / "selection.json").write_text(json.dumps(selection), encoding="utf-8")
            tampered = (reports / "selection.json").read_bytes()
            self.assertEqual(call("cv", force=True), 1)
            self.assertEqual((reports / "selection.json").read_bytes(), tampered)
            self.assertEqual(call("cv", force=True, accept_changed_selection=True), 0)

    def test_final_refuses_inconsistent_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports, call = built_workspace(Path(tmp))
            self.assertEqual(call("cv"), 0)
            original = json.loads((reports / "selection.json").read_text(encoding="utf-8"))
            for change in ("drop_primary", "features", "models", "thresholds"):
                broken = copy.deepcopy(original)
                if change == "drop_primary":
                    del broken["primary_model"]
                elif change == "features":
                    broken["features"] = broken["features"][:-1]
                elif change == "models":
                    broken["models"]["extra"] = "fit"
                else:
                    del broken["thresholds_max_f1"]["dummy"]
                (reports / "selection.json").write_text(json.dumps(broken), encoding="utf-8")
                self.assertEqual(call("final"), 1, change)
                self.assertFalse((reports / "test_metrics.json").exists(), change)

    def test_code_version_check(self):
        models = msm.build_models(MODEL_SETTINGS, 42)
        selection = {"models": {n: s[0] for n, s in models.items()}, "thresholds_max_f1": {n: 0.5 for n in models},
                     "features": FEATURES, "primary_model": {"name": "xgboost"}, "dataset_sha256": "x",
                     "git": {"commit": "abc", "dirty": False}}
        check = tms.check_selection
        self.assertEqual(check(selection, models, True, {"commit": "abc", "dirty": False}), [])
        self.assertTrue(check(selection, models, True, {"commit": "def", "dirty": False}))   # different commit
        self.assertTrue(check(selection, models, True, {"commit": "abc", "dirty": True}))    # uncommitted changes
        dirty_cv = {**selection, "git": {"commit": "abc", "dirty": True}}
        self.assertTrue(check(dirty_cv, models, True, {"commit": "abc", "dirty": False}))    # CV ran on dirty code
        self.assertEqual(check(dirty_cv, models, False, {"commit": "zzz", "dirty": True}), [])  # check disabled


REAL = PROJECT_ROOT / "data" / "processed" / "market_stress"


@unittest.skipUnless((REAL / "dataset.csv").exists(), "market-stress dataset not built")
class TestCvStageRealDevelopmentData(unittest.TestCase):
    def test_cv_uses_development_rows_only(self):
        settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
        settings["market_stress"].update({"bootstrap_resamples": 20, "bootstrap_sensitivity_blocks": []})
        before = hashlib.sha256((REAL / "dataset.csv").read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            with redirect_stdout(io.StringIO()):
                code = tms.run("cv", REAL, Path(tmp) / "r", Path(tmp) / "m", settings, False,
                               raw_dir=PROJECT_ROOT / "data" / "raw" / "market")
            self.assertEqual(code, 0)
            oof = pd.read_csv(Path(tmp) / "r" / "oof_predictions.csv")
            self.assertEqual(len(oof), 3753)
            self.assertEqual(int(oof["label"].sum()), 1157)
            self.assertLessEqual(oof["date"].max(), "2014-12-02")
            self.assertFalse((Path(tmp) / "m").exists())
        self.assertEqual(hashlib.sha256((REAL / "dataset.csv").read_bytes()).hexdigest(), before)


if __name__ == "__main__":
    unittest.main()
