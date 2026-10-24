"""Tests for chronological walk-forward splits with a purge gap (D3, D12). Synthetic dates only."""

import unittest

import numpy as np
import pandas as pd

from fincrisis.evaluation.splits import describe_splits, walk_forward_splits

DATES = pd.bdate_range("1995-01-02", "2020-12-31")
WARM_UP = 300


def eligible(n=len(DATES), warm_up=WARM_UP):
    mask = np.ones(n, dtype=bool)
    mask[:warm_up] = False
    return mask


def make(years=range(2000, 2015), test_start="2015-01-01", purge=20, horizon=20, mask=None):
    return walk_forward_splits(DATES, eligible() if mask is None else mask, list(years), test_start, purge, horizon)


class TestWalkForward(unittest.TestCase):
    def setUp(self):
        self.splits = make()
        self.positions = np.arange(len(DATES))

    def test_one_fold_per_year_in_order(self):
        self.assertEqual([f["validation_year"] for f in self.splits["folds"]], list(range(2000, 2015)))

    def test_training_ends_purge_sessions_before_validation(self):
        for fold in self.splits["folds"]:
            v0 = int(np.flatnonzero(DATES.year == fold["validation_year"])[0])
            self.assertEqual(fold["train"].max(), v0 - 21)          # 20 sessions purged
            np.testing.assert_array_equal(fold["purged"], np.arange(v0 - 20, v0))
            self.assertEqual(fold["validation"].min(), v0)
            self.assertTrue((DATES[fold["validation"]].year == fold["validation_year"]).all())
            self.assertFalse(set(fold["train"]) & set(fold["validation"]))
            # no training label (row i uses closes up to i + 20) reaches the validation block
            self.assertLess(fold["train"].max() + 20, fold["validation"].min())

    def test_training_windows_expand(self):
        starts = {int(f["train"].min()) for f in self.splits["folds"]}
        self.assertEqual(starts, {WARM_UP})
        ends = [int(f["train"].max()) for f in self.splits["folds"]]
        self.assertEqual(ends, sorted(ends))

    def test_development_and_test_are_separated_by_the_purge(self):
        t0 = int(np.flatnonzero(DATES >= "2015-01-01")[0])
        self.assertEqual(self.splits["first_test_position"], t0)
        self.assertEqual(self.splits["development"].max(), t0 - 21)
        np.testing.assert_array_equal(self.splits["test_purged"], np.arange(t0 - 20, t0))
        self.assertEqual(self.splits["test"].min(), t0)
        self.assertEqual(self.splits["test"].max(), len(DATES) - 1)
        for fold in self.splits["folds"]:
            self.assertLess(fold["validation"].max() + 20, t0)
        last = self.splits["folds"][-1]
        self.assertEqual(last["validation_rows_removed_before_test"], 20)

    def test_ineligible_rows_never_appear(self):
        mask = eligible()
        mask[3000:3010] = False
        splits = make(mask=mask)
        used = np.concatenate([splits["development"], splits["test"], *[f["train"] for f in splits["folds"]],
                               *[f["validation"] for f in splits["folds"]]])
        self.assertTrue(mask[used].all())

    def test_describe(self):
        target = (np.arange(len(DATES)) % 50 == 0).astype(float)
        summary = describe_splits(self.splits, DATES, target)
        self.assertEqual(len(summary["folds"]), 15)
        self.assertEqual(summary["test"]["first_date"], str(DATES[self.splits["test"][0]].date()))
        self.assertEqual(summary["folds"][0]["purged"]["rows"], 20)


class TestInvalidConfigurations(unittest.TestCase):
    def test_rejects_purge_shorter_than_horizon(self):
        with self.assertRaises(ValueError):
            make(purge=10, horizon=20)

    def test_rejects_validation_overlapping_test(self):
        with self.assertRaises(ValueError):
            make(years=range(2000, 2016), test_start="2015-01-01")

    def test_rejects_missing_test_rows_or_years(self):
        with self.assertRaises(ValueError):
            make(test_start="2030-01-01", years=range(2000, 2015))
        with self.assertRaises(ValueError):
            make(years=[1990])

    def test_rejects_fold_without_training_rows(self):
        with self.assertRaises(ValueError):
            make(years=[1995, 1996])  # warm-up covers 1995; nothing to train on

    def test_rejects_unsorted_dates_and_length_mismatch(self):
        with self.assertRaises(ValueError):
            walk_forward_splits(DATES[::-1], eligible(), [2000], "2015-01-01", 20, 20)
        with self.assertRaises(ValueError):
            walk_forward_splits(DATES, eligible()[:-1], [2000], "2015-01-01", 20, 20)


if __name__ == "__main__":
    unittest.main()
