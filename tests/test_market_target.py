"""Tests for the market-stress target (docs/decisions.md, D2). Synthetic data only."""

import unittest

import numpy as np
import pandas as pd

from fincrisis.features.market_target import forward_min_decline, stress_label


def series(values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range("2020-01-01", periods=len(values)))


def naive_label(c: np.ndarray, t: int, horizon: int, threshold: float) -> float:
    if t + horizon >= len(c):
        return np.nan
    worst = min(c[j] / max(c[t:j + 1]) - 1 for j in range(t + 1, t + horizon + 1))
    return float(worst <= -threshold + 1e-12)


class TestBoundaryCases(unittest.TestCase):
    def label0(self, values, horizon=1):
        return stress_label(series(values), horizon, 0.05).iloc[0]

    def test_decline_exactly_five_percent_counts(self):
        self.assertEqual(self.label0([100, 95]), 1.0)

    def test_decline_just_under_five_percent_does_not_count(self):
        self.assertEqual(self.label0([100, 95.0001]), 0.0)

    def test_recovery_after_breach_still_counts(self):
        self.assertEqual(self.label0([100, 94, 100], horizon=2), 1.0)

    def test_new_high_then_drop_counts_from_new_high(self):
        self.assertEqual(self.label0([100, 110, 104], horizon=2), 1.0)  # 104/110 - 1 = -5.45%

    def test_new_high_with_small_pullback_does_not_count(self):
        self.assertEqual(self.label0([100, 110, 106], horizon=2), 0.0)  # -3.6% from the new high

    def test_decline_measured_from_running_max_not_start(self):
        self.assertEqual(self.label0([100, 96, 94], horizon=2), 1.0)
        self.assertEqual(self.label0([100, 97, 96], horizon=2), 0.0)

    def test_rising_path_is_zero(self):
        self.assertEqual(self.label0([100, 101, 102, 103], horizon=3), 0.0)

    def test_decline_value(self):
        decline = forward_min_decline(series([100, 110, 104, 120]), 3)
        self.assertAlmostEqual(decline.iloc[0], 104 / 110 - 1)


class TestHorizon(unittest.TestCase):
    def test_drop_at_last_horizon_day_counts_and_after_it_does_not(self):
        prices = [100.0] * 21 + [94.0] + [100.0] * 30
        label = stress_label(series(prices), 20, 0.05)
        self.assertEqual(label.iloc[1], 1.0)   # day 21 is t+20 for t=1
        self.assertEqual(label.iloc[0], 0.0)   # day 21 is t+21 for t=0: outside the horizon

    def test_incomplete_horizon_is_nan_not_zero(self):
        label = stress_label(series(np.linspace(100, 80, 30)), 20, 0.05)
        self.assertTrue(label.iloc[-20:].isna().all())
        self.assertFalse(label.iloc[:-20].isna().any())

    def test_series_shorter_than_horizon_is_all_nan(self):
        self.assertTrue(stress_label(series([100, 99, 98]), 20, 0.05).isna().all())

    def test_matches_naive_implementation(self):
        rng = np.random.default_rng(0)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400)))
        label = stress_label(series(c), 20, 0.05).to_numpy()
        expected = np.array([naive_label(c, t, 20, 0.05) for t in range(len(c))])
        np.testing.assert_array_equal(label, expected)
        self.assertTrue(0 < np.nansum(label) < np.sum(~np.isnan(label)))


class TestUsesOnlyTheWindow(unittest.TestCase):
    def test_prices_before_t_and_after_t_plus_h_do_not_matter(self):
        rng = np.random.default_rng(1)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, 120)))
        base = stress_label(series(c), 20, 0.05)
        t = 50
        changed = c.copy()
        changed[:t] *= 3.0          # history before t
        changed[t + 21:] *= 0.1     # beyond the horizon
        self.assertEqual(stress_label(series(changed), 20, 0.05).iloc[t], base.iloc[t])


class TestInvalidInputs(unittest.TestCase):
    def test_rejects_bad_prices(self):
        for values in ([100, np.nan, 99], [100, 0, 99], [100, -1, 99], [100, np.inf, 99]):
            with self.assertRaises(ValueError, msg=str(values)):
                stress_label(series(values), 1, 0.05)

    def test_rejects_unsorted_or_duplicate_dates(self):
        s = series([100, 101, 102])
        with self.assertRaises(ValueError):
            stress_label(s.iloc[::-1], 1, 0.05)
        with self.assertRaises(ValueError):
            stress_label(pd.concat([s, s.iloc[[1]]]), 1, 0.05)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            stress_label(series([100, 101]), 0, 0.05)
        with self.assertRaises(ValueError):
            stress_label(series([100, 101]), 1, 1.5)


if __name__ == "__main__":
    unittest.main()
