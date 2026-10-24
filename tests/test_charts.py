"""Tests for the dashboard's Plotly chart builders (docs/decisions.md, D20).

Most tests use small hand-made inputs shaped like the loaders' outputs, so
they check structure and edge cases without data. One test builds every chart
from the real frozen outputs (skipped if absent) and checks that the numbers
shown are the recorded ones.
"""

import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fincrisis.dashboard import charts as ch
from fincrisis.dashboard import loaders as ld
from tests.test_dashboard import DIRS, PATHS, PRESENT, PROJECT_ROOT, tree_hashes

DATES = pd.bdate_range("2014-12-01", periods=40)


def snapshot(n=30):
    index = pd.bdate_range("2026-09-01", periods=n)
    return {"spx": pd.Series(np.linspace(5000, 5100, n), index=index),
            "vix": pd.Series(np.linspace(15, 20, n), index=index), "as_of": index[-1].date() if n else date(2026, 10, 23)}


def market_results():
    oof = pd.DataFrame({"date": DATES[:20], "label": [0, 0, 1, 1, 0] * 4, "rule_x": np.linspace(0.0, 0.2, 20),
                        "xgboost": np.linspace(0.2, 0.0, 20)})
    test = pd.DataFrame({"date": DATES[20:], "label": [1, 1, 0, 0, 0] * 4, "rule_x": np.linspace(0.05, 0.15, 20),
                         "xgboost": np.linspace(0.1, 0.3, 20)})
    per_year = [{"year": 2015, "average_precision": 0.4, "prevalence": 0.2, "status": "ok", "label_1": 50},
                {"year": 2016, "average_precision": 0.1, "prevalence": 0.05, "status": "few_positives", "label_1": 3}]
    at = {"threshold": 0.1, "precision": 0.3, "recall": 0.6, "tn": 5, "fp": 7, "fn": 2, "tp": 6}
    models = {"rule_x": {"average_precision": 0.36, "per_year": per_year, "calibration": None, "at_max_f1_threshold": at},
              "xgboost": {"average_precision": 0.33, "per_year": per_year, "at_max_f1_threshold": {**at, "threshold": 0.2},
                          "calibration": [{"bin": "0.0-0.1", "n": 10, "mean_predicted": 0.05, "observed_rate": 0.2},
                                          {"bin": "0.1-0.2", "n": 8, "mean_predicted": 0.15, "observed_rate": 0.1}]}}
    bootstrap = {"primary": {"average_precision": {"rule_x": [0.25, 0.47], "xgboost": [0.2, 0.45]}}}
    return {"oof_predictions": oof, "test_predictions": test, "thresholds": {"rule_x": 0.1, "xgboost": 0.2},
            "primary_model": "rule_x",
            "test_metrics": {"models": models, "prevalence": 0.212, "rows": 20, "bootstrap": bootstrap},
            "cv_metrics": {"models": models, "prevalence": 0.3, "rows": 20}}


def distress_results():
    at = {"threshold": 0.4, "tn": 90, "fp": 5, "fn": 2, "tp": 3, "precision": 0.375, "recall": 0.6}
    families = {"dummy": {"config": "dummy", "average_precision": 0.05, "roc_auc": 0.5, "at_max_f1_threshold": at,
                          "at_0.5": {**at, "threshold": 0.5}},
                "xgboost": {"config": "xgb_cfg", "average_precision": 0.49, "roc_auc": 0.95, "at_max_f1_threshold": at,
                            "at_0.5": {**at, "threshold": 0.5},
                            "bootstrap_95ci_at_max_f1": {"average_precision": [0.36, 0.64], "roc_auc": [0.93, 0.97]}}}
    predictions = pd.DataFrame({"row_id": range(10), "label": [0, 1] * 5, "dummy": [0.05] * 10,
                                "xgb_cfg": np.linspace(0, 1, 10)})
    cv = pd.DataFrame({"config": ["xgb_cfg"] * 3 + ["other"] * 3, "average_precision": [0.4, 0.5, 0.45, 0.3, 0.2, 0.25]})
    importance = pd.DataFrame({"config": ["xgb_cfg"] * 2, "fold": [0, 1], "mean_drop": [0.2, 0.1],
                               "base_average_precision": [0.5, 0.4]})
    return {"test_metrics": {"results": families, "test_rows": 100, "test_label_1": 5}, "test_predictions": predictions,
            "cv_results": cv, "permutation_importance": importance, "selection": {"large_value_columns": ["a", "b", "c"]}}


class TestHelpers(unittest.TestCase):
    def test_episodes(self):
        days = pd.Series(pd.bdate_range("2020-01-01", periods=6))
        self.assertEqual(ch._episodes(days, [1, 1, 0, 0, 1, 1]), [(days[0], days[1]), (days[4], days[5])])
        self.assertEqual(ch._episodes(days, [0] * 6), [])

    def test_break_gaps_inserts_an_empty_row(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-03-01"]), "v": [1.0, 2.0, 3.0]})
        broken = ch._break_gaps(frame, "date")
        self.assertEqual(len(broken), 4)
        self.assertTrue(np.isnan(broken["v"].iloc[2]))

    def test_build_or_message_explains_missing_input(self):
        fig = ch.build_or_message(ch.price_and_vix, None)
        self.assertEqual(len(fig.data), 0)
        self.assertIn("market snapshot is missing", fig.layout.annotations[0].text)
        self.assertIn("unavailable", fig.layout.title.text)


class TestMarketCharts(unittest.TestCase):
    def test_price_and_vix_uses_two_panels_not_two_axes(self):
        snap = snapshot()
        fig = ch.price_and_vix(snap)
        self.assertEqual([t.name for t in fig.data], ["S&P 500 close", "VIX close"])
        self.assertEqual((fig.data[0].yaxis, fig.data[1].yaxis), ("y", "y2"))
        self.assertEqual(fig.layout.yaxis.type, "log")
        self.assertIsNone(fig.layout.yaxis.overlaying)
        self.assertIn(f"to {snap['as_of']}", fig.layout.title.text)
        for bad in (snapshot(0), {"spx": snapshot()["spx"]}):
            with self.assertRaises(ch.ChartDataError):
                ch.price_and_vix(bad)

    def test_drawdown_and_volatility(self):
        features = pd.DataFrame({"drawdown_252": [np.nan, -0.1, -0.2], "downside_vol_20": [np.nan, 0.1, 0.3]},
                                index=pd.bdate_range("2026-01-01", periods=3))
        fig = ch.drawdown_and_volatility(features)
        self.assertEqual(len(fig.data[0].x), 2)  # warm-up rows are not drawn
        with self.assertRaises(ch.ChartDataError):
            ch.drawdown_and_volatility(features.drop(columns="drawdown_252"))
        with self.assertRaises(ch.ChartDataError):
            ch.drawdown_and_volatility(features.iloc[:1])

    def test_score_timeline_separates_observations_scores_and_signals(self):
        results = market_results()
        fig = ch.stress_score_timeline(results)
        names = [t.name for t in fig.data]
        self.assertIn("Observed label-1 period (known afterwards)", names)
        self.assertIn("rule_x score, held-out test", names)
        self.assertIn("Signal: score ≥ frozen threshold", names)
        data = pd.concat([results["oof_predictions"], results["test_predictions"]])
        signal = next(t for t in fig.data if t.name.startswith("Signal"))
        self.assertEqual(len(signal.x), int((data["rule_x"] >= 0.1).sum()))
        episodes = [s for s in fig.layout.shapes if s.type == "rect"]
        self.assertEqual(len(episodes), len(ch._episodes(data["date"], data["label"])))
        threshold = [s for s in fig.layout.shapes if s.type == "line" and s.y0 == s.y1]
        self.assertEqual(threshold[0].y0, 0.1)
        self.assertIn("Frozen threshold 0.1000", names)
        self.assertEqual(fig.layout.xaxis.type, "date")
        self.assertIn("not a probability", fig.layout.yaxis.title.text)
        with self.assertRaises(ch.ChartDataError):
            ch.stress_score_timeline(results, model="unknown")
        with self.assertRaises(ch.ChartDataError):
            ch.stress_score_timeline({**results, "test_predictions": results["test_predictions"].drop(columns="label")})

    def test_current_signal_marker_only_for_the_matching_model(self):
        current = {"as_of": "2015-02-06", "primary": {"name": "rule_x", "score": 0.12, "elevated": True}}
        fig = ch.stress_score_timeline(market_results(), current=current)
        latest = next(t for t in fig.data if t.name.startswith("Latest snapshot day, 2015-02-06 (signal"))
        self.assertEqual(list(latest.text), ["Latest: signal"])
        other = ch.stress_score_timeline(market_results(), model="xgboost", current=current)
        self.assertFalse(any(t.name.startswith("Latest snapshot day") for t in other.data))

    def test_model_metrics_show_recorded_values_and_intervals(self):
        fig = ch.stress_model_metrics(market_results())
        bar = fig.data[0]
        self.assertEqual(list(bar.y), [0.36, 0.33])
        self.assertEqual(list(bar.x)[0], "rule_x<br>(primary)")
        self.assertIn("Base rate 0.212 (expected AP of a random score)", [trace.name for trace in fig.data])
        self.assertAlmostEqual(bar.error_y.arrayminus[0], 0.36 - 0.25)
        self.assertAlmostEqual(bar.error_y.array[0], 0.47 - 0.36)
        self.assertEqual(fig.layout.shapes[0].y0, 0.212)
        validation = ch.stress_model_metrics(market_results(), "validation")  # no bootstrap: no error bars
        self.assertEqual(list(validation.data[0].error_y.array), [0, 0])
        with self.assertRaises(ch.ChartDataError):
            ch.stress_model_metrics(market_results(), "train")

    def test_per_year_marks_unreliable_years(self):
        fig = ch.stress_per_year(market_results())
        rule = fig.data[0]
        self.assertEqual(rule.marker.color[0], ch.color("rule_x"))
        self.assertEqual(rule.marker.color[1], ch.SURFACE)  # hollow: status is not "ok"
        self.assertEqual(fig.data[-1].name, "Base rate (share of label-1 days)")
        results = market_results()
        results["test_metrics"]["models"]["rule_x"] = {**results["test_metrics"]["models"]["rule_x"], "per_year": []}
        with self.assertRaises(ch.ChartDataError):
            ch.stress_per_year(results)

    def test_calibration_skips_rule_scores(self):
        fig = ch.stress_calibration(market_results())
        self.assertEqual([t.name for t in fig.data], ["Perfect calibration", "XGBoost"])
        results = market_results()
        results["test_metrics"]["models"]["xgboost"] = {**results["test_metrics"]["models"]["xgboost"], "calibration": None}
        with self.assertRaises(ch.ChartDataError):
            ch.stress_calibration(results)

    def test_precision_recall_marks_the_recorded_operating_point(self):
        fig = ch.stress_precision_recall(market_results())
        points = [t for t in fig.data if t.mode == "markers"]
        self.assertEqual([(p.x[0], p.y[0]) for p in points], [(0.6, 0.3), (0.6, 0.3)])
        self.assertIn("AP 0.360", fig.data[0].name)
        one_class = market_results()
        one_class["test_predictions"] = one_class["test_predictions"].assign(label=0)
        with self.assertRaises(ch.ChartDataError):
            ch.stress_precision_recall(one_class)


class TestConfusionMatrix(unittest.TestCase):
    def test_counts_and_row_shares(self):
        fig = ch.confusion_matrix({"tn": 8, "fp": 2, "fn": 1, "tp": 3}, "Test")
        heat = fig.data[0]
        np.testing.assert_allclose(heat.z, [[0.8, 0.2], [0.25, 0.75]])
        self.assertEqual(heat.text[1][1], "<b>3</b><br>75% of row")

    def test_empty_class_and_missing_counts(self):
        fig = ch.confusion_matrix({"tn": 0, "fp": 0, "fn": 1, "tp": 3}, "Test")
        np.testing.assert_allclose(fig.data[0].z[0], [0, 0])
        with self.assertRaises(ch.ChartDataError):
            ch.confusion_matrix({"tn": 1}, "Test")

    def test_stress_confusion_uses_the_frozen_threshold(self):
        fig = ch.stress_confusion(market_results())
        self.assertIn("0.1000", fig.layout.title.text)
        with self.assertRaises(ch.ChartDataError):
            ch.stress_confusion(market_results(), model="nope")


class TestDistressCharts(unittest.TestCase):
    def test_model_metrics_two_panels_with_intervals(self):
        fig = ch.distress_model_metrics(distress_results())
        ap, auc = [trace for trace in fig.data if trace.type == "bar"]
        self.assertEqual(list(ap.y), [0.05, 0.49])
        self.assertAlmostEqual(ap.error_y.arrayminus[1], 0.49 - 0.36)
        self.assertEqual(list(auc.y), [0.5, 0.95])
        self.assertEqual((ap.xaxis, auc.xaxis), ("x", "x2"))
        self.assertIn("label meaning unverified", fig.layout.title.text)
        self.assertEqual(fig.layout.shapes[0].y0, 0.05)  # base rate 5 / 100

    def test_confusion_operating_points(self):
        self.assertIn("0.400", ch.distress_confusion(distress_results()).layout.title.text)
        self.assertIn("At 0.5", ch.distress_confusion(distress_results(), at="0.5").layout.title.text)
        for kwargs in ({"family": "nope"}, {"at": "0.3"}):
            with self.assertRaises(ch.ChartDataError):
                ch.distress_confusion(distress_results(), **kwargs)

    def test_precision_recall_uses_selected_configs(self):
        fig = ch.distress_precision_recall(distress_results())
        self.assertEqual(fig.data[0].name, "XGBoost (AP 0.490)")

    def test_cv_folds_highlight_selected_configs(self):
        fig = ch.distress_cv_folds(distress_results())
        selected = next(t for t in fig.data if t.name.startswith("xgb_cfg"))
        self.assertEqual(selected.marker.color, ch.color("xgboost"))
        other = next(t for t in fig.data if t.name == "other")
        self.assertEqual(other.marker.color, ch.REFERENCE)

    def test_group_importance_is_labelled_as_grouped(self):
        fig = ch.distress_group_importance(distress_results())
        self.assertIn("3 large-value columns", fig.layout.title.text)
        self.assertIn("not per feature", fig.layout.title.text)
        without = ch.distress_group_importance({**distress_results(), "selection": None})
        self.assertIn("the large-value columns", without.layout.title.text)
        with self.assertRaises(ch.ChartDataError):
            ch.distress_group_importance({**distress_results(), "permutation_importance": pd.DataFrame()})


@unittest.skipUnless(PRESENT, "frozen market/distress outputs not present")
class TestChartsOnFrozenOutputs(unittest.TestCase):
    def test_every_chart_builds_and_shows_recorded_numbers(self):
        settings = ld.load_settings()
        before = {k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}
        snap = ld.market_snapshot(PROJECT_ROOT, PATHS)
        features, _ = ld.market_features(snap)
        stress = ld.market_stress_results(PROJECT_ROOT, PATHS)
        distress = ld.distress_results(PROJECT_ROOT, PATHS)
        current = ld.current_market_signal(PROJECT_ROOT, PATHS, settings["agent"]["stale_after_days"])
        figures = [ch.price_and_vix(snap), ch.drawdown_and_volatility(features),
                   ch.stress_score_timeline(stress, current=current), ch.stress_model_metrics(stress),
                   ch.stress_model_metrics(stress, "validation"), ch.stress_per_year(stress),
                   ch.stress_per_year(stress, "validation"), ch.stress_calibration(stress), ch.stress_confusion(stress),
                   ch.stress_precision_recall(stress), ch.distress_model_metrics(distress),
                   ch.distress_confusion(distress), ch.distress_precision_recall(distress),
                   ch.distress_cv_folds(distress), ch.distress_group_importance(distress)]
        self.assertTrue(all(f.data for f in figures))
        test = stress["test_metrics"]
        self.assertEqual(list(figures[3].data[0].y), [m["average_precision"] for m in test["models"].values()])
        primary = stress["primary_model"]
        at = test["models"][primary]["at_max_f1_threshold"]
        np.testing.assert_array_equal(figures[8].data[0].z.sum(), 2.0)  # two rows of shares
        self.assertIn(f"{at['threshold']:.4f}", figures[8].layout.title.text)
        results = distress["test_metrics"]["results"]
        self.assertEqual(list(figures[10].data[0].y), [r["average_precision"] for r in results.values()])
        self.assertIn(str(len(distress["selection"]["large_value_columns"])), figures[14].layout.title.text)
        self.assertEqual({k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}, before)


if __name__ == "__main__":
    unittest.main()
