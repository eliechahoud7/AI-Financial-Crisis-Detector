"""Tests for the market data snapshot (docs/decisions.md, D10). No network access:
downloads are replaced by synthetic frames shaped like yfinance output."""

import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fincrisis.data import market

PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_market_data", PROJECT_ROOT / "scripts" / "download_market_data.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)

START, END = date(2020, 1, 1), date(2020, 12, 31)
SYMBOLS = {"gspc": "^GSPC", "vix": "^VIX"}


def synthetic(start=START, end=END, seed=0, level=3000.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    close = level * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    frame = pd.DataFrame({"Open": close * 0.999, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                          "Adj Close": close, "Volume": rng.integers(1_000, 5_000, len(dates))}, index=dates)
    frame.index.name = "Date"
    return frame


def fake_fetch(frames: dict):
    calls = []

    def fetch(symbol, start, end_inclusive):
        calls.append(symbol)
        value = frames[symbol]
        if isinstance(value, Exception):
            raise value
        return value.copy()
    fetch.calls = calls
    return fetch


def run_quiet(*args, **kwargs) -> int:
    with redirect_stdout(io.StringIO()):
        return dl.run(*args, **kwargs)


class TestNormalize(unittest.TestCase):
    def test_flattens_multiindex_and_drops_timezone(self):
        frame = synthetic()
        frame.columns = pd.MultiIndex.from_product([frame.columns, ["^GSPC"]])
        frame.index = frame.index.tz_localize("America/New_York")
        out = market.normalize_download(frame)
        self.assertEqual(list(out.columns), market.REQUIRED_COLUMNS)
        self.assertIsNone(out.index.tz)

    def test_rejects_empty_and_missing_columns(self):
        with self.assertRaises(market.MarketDataError):
            market.normalize_download(pd.DataFrame())
        with self.assertRaises(market.MarketDataError):
            market.normalize_download(synthetic().drop(columns="Close"))


class TestValidation(unittest.TestCase):
    def check(self, frame):
        return market.validate_series(market.normalize_download(frame), "^T", START, END)

    def test_clean_series_has_no_errors(self):
        summary = self.check(synthetic())
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["rows"], len(pd.bdate_range(START, END)))

    def test_errors(self):
        frame = synthetic()
        cases = {
            "duplicate": pd.concat([frame, frame.iloc[[5]]]).sort_index(),
            "missing close": frame.assign(Close=frame["Close"].where(np.arange(len(frame)) != 10)),
            "non-positive close": frame.assign(Close=frame["Close"].where(np.arange(len(frame)) != 10, 0.0)),
            "late start": frame[frame.index >= "2020-03-01"],
            "stale end": frame[frame.index <= "2020-11-30"],
            "beyond end": pd.concat([frame, synthetic(date(2021, 1, 4), date(2021, 1, 4))]),
            "low coverage": frame.iloc[::2],
        }
        for name, case in cases.items():
            self.assertTrue(self.check(case)["errors"], name)

    def test_gap_is_a_warning_not_an_error(self):
        frame = synthetic()
        gapped = frame[(frame.index < "2020-09-10") | (frame.index > "2020-09-16")]
        summary = self.check(gapped)
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["gaps"], [{"after": "2020-09-09", "before": "2020-09-17", "calendar_days": 8}])

    def test_calendar_comparison(self):
        a, b = synthetic(), synthetic().drop(index=pd.Timestamp("2020-06-01"))
        result = market.compare_calendars(a, b, "gspc", "vix")
        self.assertEqual((result["dates_only_in_gspc"], result["dates_only_in_vix"]), (1, 0))
        self.assertEqual(result["examples_only_in_gspc"], ["2020-06-01"])

    def test_csv_round_trip(self):
        frame = market.normalize_download(synthetic())
        text = market.to_csv_text(frame)
        parsed = pd.read_csv(io.StringIO(text), index_col="Date", parse_dates=["Date"], float_precision="round_trip")
        np.testing.assert_array_equal(parsed.to_numpy(dtype=float), frame.to_numpy(dtype=float))


class TestRange(unittest.TestCase):
    def test_null_end_uses_day_before_today(self):
        start, end, _ = dl.resolve_range({"start_date": "1990-01-01", "end_date": None}, date(2026, 10, 24))
        self.assertEqual((start, end), (date(1990, 1, 1), date(2026, 10, 23)))

    def test_configured_end_is_used(self):
        _, end, source = dl.resolve_range({"start_date": "1990-01-01", "end_date": "2026-10-23"}, date(2027, 1, 1))
        self.assertEqual(end, date(2026, 10, 23))
        self.assertIn("config", source)


class TestSnapshot(unittest.TestCase):
    def test_writes_files_metadata_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "market"
            vix = synthetic(seed=1, level=20.0).drop(index=pd.Timestamp("2020-06-01"))
            fetch = fake_fetch({"^GSPC": synthetic(), "^VIX": vix})
            self.assertEqual(run_quiet(out, SYMBOLS, START, END, "test", False, fetch), 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["gspc.csv", "metadata.json", "vix.csv"])
            meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            for name in ("gspc", "vix"):
                s = meta["series"][name]
                self.assertEqual(hashlib.sha256((out / s["file"]).read_bytes()).hexdigest(), s["sha256"])
                self.assertEqual(s["columns"], market.REQUIRED_COLUMNS)
            self.assertEqual(meta["requested"]["start"], "2020-01-01")
            self.assertEqual(meta["requested"]["end_inclusive"], "2020-12-31")
            self.assertEqual(meta["last_completed_trading_date_in_all_series"], "2020-12-31")
            self.assertEqual(meta["calendar_comparison"]["dates_only_in_gspc"], 1)
            self.assertIn("yfinance", meta["versions"])
            self.assertTrue(meta["downloaded_at_utc"].endswith("+00:00"))

    def test_no_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "market"
            frames = {"^GSPC": synthetic(), "^VIX": synthetic(seed=1, level=20.0)}
            self.assertEqual(run_quiet(out, SYMBOLS, START, END, "test", False, fake_fetch(frames)), 0)
            before = {p.name: p.read_bytes() for p in out.iterdir()}
            second = fake_fetch({"^GSPC": synthetic(seed=5), "^VIX": synthetic(seed=6, level=20.0)})
            self.assertEqual(run_quiet(out, SYMBOLS, START, END, "test", False, second), 1)
            self.assertEqual(second.calls, [])  # refused before downloading
            self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, before)
            self.assertEqual(run_quiet(out, SYMBOLS, START, END, "test", True, second), 0)
            self.assertNotEqual((out / "gspc.csv").read_bytes(), before["gspc.csv"])

    def test_failed_validation_or_download_writes_nothing(self):
        bad_cases = [
            {"^GSPC": synthetic(), "^VIX": synthetic(seed=1).iloc[:50]},
            {"^GSPC": synthetic(), "^VIX": ConnectionError("network unavailable")},
            {"^GSPC": pd.DataFrame(), "^VIX": synthetic(seed=1)},
        ]
        for frames in bad_cases:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "market"
                self.assertEqual(run_quiet(out, SYMBOLS, START, END, "test", False, fake_fetch(frames)), 1)
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
