"""Tests for the reproducibility (drift) checks (docs/decisions.md, D23).

The checks run read-only on the project's frozen artifacts (skipped if absent).
Each kind of drift is simulated in memory (changed feature or target code, a
model that scores differently, a different scoring path), never by changing a
file, and must be reported.
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

from fincrisis.agents import tools
from fincrisis.data import taiwan_bankruptcy as tb
from fincrisis.dashboard import loaders as ld
from fincrisis.features import market_features as mf
from fincrisis.qa import drift
from tests.test_dashboard import PRESENT, PROJECT_ROOT
from tests.test_integrity import ALL_PRESENT

SETTINGS = ld.load_settings()


def failed(checks):
    return {c.name for c in checks if not c.ok}


class ShiftedModel:
    """A saved model whose scores moved by `shift`, as a library upgrade or a changed model file could cause."""

    def __init__(self, model, shift):
        self.model, self.shift = model, shift

    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        return np.column_stack([p[:, 0] - self.shift, p[:, 1] + self.shift])


def shifted_models(shift):
    original = tools.load_verified_model
    return unittest.mock.patch.object(tools, "load_verified_model",
                                      lambda directory, name: ShiftedModel(original(directory, name), shift))


class TestDifference(unittest.TestCase):
    def test_tolerance_shape_and_missing_values(self):
        self.assertTrue(drift._difference("g", "n", [1.0, 2.0], [1.0, 2.0 + 1e-9], 1e-6).ok)
        self.assertEqual(drift._difference("g", "n", [1.0, 2.0], [1.0, 2.1], 1e-6).status, "mismatch")
        self.assertIn("shape", drift._difference("g", "n", [1.0], [1.0, 2.0], 0).detail)
        self.assertIn("missing values", drift._difference("g", "n", [np.nan, 1.0], [0.0, 1.0], 0).detail)
        self.assertTrue(drift._difference("g", "n", [np.nan, 1.0], [np.nan, 1.0], 0).ok)


@unittest.skipUnless(ALL_PRESENT, "frozen inputs and outputs not present")
class TestCurrentCodeReproducesFrozenResults(unittest.TestCase):
    def test_everything_reproduces(self):
        checks = drift.verify_all(PROJECT_ROOT, SETTINGS)
        self.assertEqual(failed(checks), set(), [c for c in checks if not c.ok])
        self.assertEqual({c.group for c in checks},
                         {"market dataset", "market models", "distress models", "latest signal"})
        self.assertEqual(sum(c.name.startswith("feature ") for c in checks), len(mf.FEATURES))


@unittest.skipUnless(PRESENT, "frozen market outputs not present")
class TestMarketDrift(unittest.TestCase):
    def test_changed_feature_code_is_detected(self):
        original = mf.build_features

        def changed(spx, vix):
            features, report = original(spx, vix)
            features["downside_vol_20"] = features["downside_vol_20"] * (1 + 1e-12)  # even a tiny change
            return features, report
        with unittest.mock.patch.object(mf, "build_features", changed):
            checks = drift.market_dataset_checks(PROJECT_ROOT, SETTINGS)
        self.assertEqual(failed(checks), {"feature downside_vol_20"})

    def test_changed_target_code_is_detected(self):
        original = drift.stress_label
        with unittest.mock.patch.object(drift, "stress_label",
                                        lambda close, horizon, threshold: original(close, horizon, threshold * 2)):
            self.assertEqual(failed(drift.market_dataset_checks(PROJECT_ROOT, SETTINGS)), {"target"})

    def test_models_that_score_differently_are_detected(self):
        with shifted_models(1e-4):
            found = failed(drift.market_model_checks(PROJECT_ROOT, SETTINGS))
        self.assertEqual(found, {"dummy test scores", "logistic_regression test scores", "xgboost test scores"})
        with shifted_models(1e-8):  # float32 rounding is not drift
            self.assertEqual(failed(drift.market_model_checks(PROJECT_ROOT, SETTINGS)), set())

    def test_a_different_scoring_path_for_the_latest_day_is_detected(self):
        original = ld.current_market_signal
        primary = ld.market_stress_results(PROJECT_ROOT, SETTINGS["paths"])["primary_model"]

        def diverging(*args, **kwargs):
            signal = original(*args, **kwargs)
            signal["primary"] = {**signal["primary"], "score": signal["primary"]["score"] * 1.01}
            return signal
        with unittest.mock.patch.object(ld, "current_market_signal", diverging):
            found = failed(drift.latest_signal_checks(PROJECT_ROOT, SETTINGS))
        self.assertEqual(found, {f"primary score ({primary})"})


@unittest.skipUnless(ALL_PRESENT, "frozen distress inputs and outputs not present")
class TestDistressDrift(unittest.TestCase):
    def test_models_that_score_differently_are_detected(self):
        with shifted_models(1e-4):
            found = failed(drift.distress_model_checks(PROJECT_ROOT, SETTINGS))
        self.assertEqual(found, {"dummy test scores", "lr_quantile test scores", "xgb_large_as_missing test scores"})

    def test_different_test_rows_are_detected(self):
        original = tb.load_processed

        def reordered(directory, subset):
            X, y, ids = original(directory, subset)
            return X.iloc[::-1], y.iloc[::-1], ids.iloc[::-1].reset_index(drop=True)
        with unittest.mock.patch.object(tb, "load_processed", reordered):
            checks = drift.distress_model_checks(PROJECT_ROOT, SETTINGS)
        self.assertEqual([(c.name, c.status) for c in checks], [("test rows", "mismatch")])


class TestMissingInputs(unittest.TestCase):
    def test_missing_artifacts_are_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as empty:
            for check in (drift.market_dataset_checks, drift.market_model_checks, drift.distress_model_checks,
                          drift.latest_signal_checks):
                result = check(Path(empty), SETTINGS)
                self.assertEqual([c.status for c in result], ["missing"], check.__name__)


@unittest.skipUnless(ALL_PRESENT, "frozen inputs and outputs not present")
class TestScript(unittest.TestCase):
    def test_cli_passes_on_the_project(self):
        import json
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "check_reproducibility.py"), "--json"],
                             capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])
        self.assertTrue(all(c["status"] == "ok" for c in json.loads(out.stdout)))


if __name__ == "__main__":
    unittest.main()
