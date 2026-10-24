"""Tests for market features and the D11 VIX alignment rule. Synthetic data only."""

import unittest

import numpy as np
import pandas as pd

from fincrisis.features import market_features as mf

FIRST_VALID = {  # first position with a value, given complete inputs
    "ret_1": 1, "ret_5": 5, "ret_20": 20, "rv_5": 5, "rv_ratio_20_60": 60, "downside_vol_20": 20,
    "drawdown_252": 251, "max_drawdown_20": 19, "max_drawdown_60": 59, "dist_ma_50": 49, "ma_50_slope": 54,
    "vix_lag1": 1, "vix_change_5": 6, "vix_change_20": 21, "vix_pct_252": 252, "vix_rv_ratio": 21,
}


def make_inputs(n: int = 400, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2000-01-03", periods=n)
    spx = pd.Series(1000 * np.exp(np.cumsum(rng.normal(0, 0.012, n))), index=dates, name="spx")
    vix = pd.Series(20 * np.exp(np.cumsum(rng.normal(0, 0.05, n))), index=dates, name="vix")
    return spx, vix


class TestWindowsAndWarmUp(unittest.TestCase):
    def test_first_valid_position_of_every_feature(self):
        features, _ = mf.build_features(*make_inputs())
        self.assertEqual(list(features.columns), mf.FEATURES)
        for name, first in FIRST_VALID.items():
            column = features[name]
            self.assertTrue(column.iloc[:first].isna().all(), f"{name} has a value before position {first}")
            self.assertFalse(column.iloc[first:].isna().any(), f"{name} has a gap after position {first}")

    def test_missingness_report(self):
        features, _ = mf.build_features(*make_inputs())
        report = mf.missingness(features)
        self.assertEqual(report["drawdown_252"], {"missing": 251, "warm_up": 251, "after_warm_up": 0})


class TestValues(unittest.TestCase):
    def setUp(self):
        self.spx, self.vix = make_inputs()
        self.features, _ = mf.build_features(self.spx, self.vix)
        self.c = self.spx.to_numpy()
        self.r = np.log(self.c[1:] / self.c[:-1])  # r[k-1] is the return ending at position k

    def test_returns(self):
        t = 100
        for k in (1, 5, 20):
            self.assertAlmostEqual(self.features[f"ret_{k}"].iloc[t], np.log(self.c[t] / self.c[t - k]))

    def test_volatility(self):
        t = 150
        last = lambda w: self.r[t - w:t]  # returns ending at positions t-w+1 .. t
        self.assertAlmostEqual(self.features["rv_5"].iloc[t], np.std(last(5), ddof=1) * np.sqrt(252))
        ratio = np.std(last(20), ddof=1) / np.std(last(60), ddof=1)
        self.assertAlmostEqual(self.features["rv_ratio_20_60"].iloc[t], ratio)
        downside = np.sqrt(np.mean(np.minimum(last(20), 0) ** 2)) * np.sqrt(252)
        self.assertAlmostEqual(self.features["downside_vol_20"].iloc[t], downside)

    def test_trend_features(self):
        t = 300
        self.assertAlmostEqual(self.features["drawdown_252"].iloc[t], self.c[t] / self.c[t - 251:t + 1].max() - 1)
        sma = lambda p: self.c[p - 49:p + 1].mean()
        self.assertAlmostEqual(self.features["dist_ma_50"].iloc[t], self.c[t] / sma(t) - 1)
        self.assertAlmostEqual(self.features["ma_50_slope"].iloc[t], sma(t) / sma(t - 5) - 1)

    def test_max_drawdown_on_known_path(self):
        close = pd.Series([100.0, 120, 90, 110, 130], index=pd.bdate_range("2020-01-01", periods=5))
        self.assertAlmostEqual(mf.max_drawdown(close, 5).iloc[-1], 90 / 120 - 1)
        self.assertAlmostEqual(mf.max_drawdown(close, 3).iloc[-1], 0.0)  # window 90, 110, 130 only rises
        self.assertAlmostEqual(mf.max_drawdown(close, 4).iloc[-1], 90 / 120 - 1)

    def test_vix_features(self):
        t = 300
        v = self.vix.to_numpy()
        lag = lambda p: v[p - 1]  # previous S&P session's VIX close
        self.assertEqual(self.features["vix_lag1"].iloc[t], lag(t))
        self.assertAlmostEqual(self.features["vix_change_5"].iloc[t], np.log(lag(t) / lag(t - 5)))
        self.assertAlmostEqual(self.features["vix_change_20"].iloc[t], np.log(lag(t) / lag(t - 20)))
        window = v[t - 252:t]  # vix_lag1 at positions t-251 .. t
        self.assertAlmostEqual(self.features["vix_pct_252"].iloc[t], np.mean(window <= lag(t)))
        rv20_previous = np.std(self.r[t - 21:t - 1], ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(self.features["vix_rv_ratio"].iloc[t], lag(t) / (100 * rv20_previous))

    def test_rolling_percentile_requires_full_window_without_gaps(self):
        s = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0, 5.0], index=pd.bdate_range("2020-01-01", periods=6))
        out = mf.rolling_percentile(s, 3)
        self.assertTrue(out.iloc[:5].isna().all())
        self.assertEqual(out.iloc[5], 1.0)


class TestNoLookAhead(unittest.TestCase):
    def test_features_at_t_ignore_all_later_data(self):
        spx, vix = make_inputs(n=420, seed=3)
        full, _ = mf.build_features(spx, vix)
        for t in (260, 300, 419):
            truncated, _ = mf.build_features(spx.iloc[:t + 1], vix.iloc[:t + 1])
            pd.testing.assert_series_equal(truncated.iloc[t], full.iloc[t], check_names=False)

    def test_vix_on_or_after_t_never_affects_features_at_t(self):
        spx, vix = make_inputs(n=420, seed=4)
        t = 350
        base, _ = mf.build_features(spx, vix)
        changed = vix.copy()
        changed.iloc[t:] *= 5.0  # VIX dated t and later
        altered, _ = mf.build_features(spx, changed)
        pd.testing.assert_series_equal(altered.iloc[t], base.iloc[t], check_names=False)
        self.assertNotEqual(altered["vix_lag1"].iloc[t + 1], base["vix_lag1"].iloc[t + 1])


class TestVixAlignmentD11(unittest.TestCase):
    def setUp(self):
        self.spx_dates = pd.DatetimeIndex(["2026-05-21", "2026-05-22", "2026-05-26", "2026-05-27"])

    def test_vix_only_date_cannot_shift_the_lag(self):
        vix = pd.Series([10.0, 11.0, 99.0, 12.0, 13.0],
                        index=pd.DatetimeIndex(["2026-05-21", "2026-05-22", "2026-05-25", "2026-05-26", "2026-05-27"]))
        lagged, report = mf.align_vix_previous_session(self.spx_dates, vix)
        self.assertTrue(np.isnan(lagged.iloc[0]))                      # first S&P date: no previous session
        self.assertEqual(lagged.loc["2026-05-26"], 11.0)              # previous S&P session is 05-22, not 05-25
        self.assertEqual(lagged.loc["2026-05-27"], 12.0)
        self.assertNotIn(99.0, lagged.to_numpy())
        self.assertEqual(report["vix_only_dates_ignored"], ["2026-05-25"])

    def test_missing_previous_session_stays_missing(self):
        vix = pd.Series([10.0, 12.0, 13.0], index=pd.DatetimeIndex(["2026-05-21", "2026-05-26", "2026-05-27"]))
        lagged, report = mf.align_vix_previous_session(self.spx_dates, vix)
        self.assertTrue(np.isnan(lagged.loc["2026-05-26"]))  # 05-22 has no VIX close; not filled from 05-21 or 05-26
        self.assertEqual(lagged.loc["2026-05-27"], 12.0)
        self.assertEqual(report["spx_dates_without_vix_close"], ["2026-05-22"])
        self.assertEqual(report["vix_lag1_missing"], 2)

    def test_lagged_value_is_always_dated_before_t(self):
        spx, vix = make_inputs(n=60)
        lagged, _ = mf.align_vix_previous_session(spx.index, vix)
        for i in range(1, len(spx)):
            self.assertEqual(lagged.iloc[i], vix.loc[spx.index[i - 1]])


class TestInvalidInputs(unittest.TestCase):
    def test_rejects_bad_prices_and_vix(self):
        spx, vix = make_inputs(n=50)
        bad = spx.copy()
        bad.iloc[10] = np.nan
        with self.assertRaises(ValueError):
            mf.build_features(bad, vix)
        bad_vix = vix.copy()
        bad_vix.iloc[10] = -1.0
        with self.assertRaises(ValueError):
            mf.build_features(spx, bad_vix)
        with self.assertRaises(ValueError):
            mf.align_vix_previous_session(spx.index, pd.concat([vix, vix.iloc[[3]]]))


if __name__ == "__main__":
    unittest.main()
