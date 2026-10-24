"""Tests for scripts/build_market_dataset.py on a synthetic snapshot (no network, no real data)."""

import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_market_dataset", PROJECT_ROOT / "scripts" / "build_market_dataset.py")
bmd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bmd)

SETTINGS = {
    "market": {"end_date": "2005-12-30"},
    "market_stress_target": {"horizon_trading_days": 20, "drawdown_threshold": 0.05, "peak_definition": "forward_running_max"},
    "evaluation": {"purge_gap_trading_days": 20},
    "market_stress": {"validation_first_year": 1997, "validation_last_year": 2001, "test_start": "2002-01-01"},
}


def write_snapshot(raw: Path, vix_extra_date=None, spx_nan_at=None) -> None:
    raw.mkdir(parents=True)
    dates = pd.bdate_range("1995-01-02", "2005-12-30")
    rng = np.random.default_rng(0)
    close = 500 * np.exp(np.cumsum(rng.normal(0, 0.013, len(dates))))
    if spx_nan_at is not None:
        close[spx_nan_at] = np.nan
    vix_close = 20 * np.exp(np.cumsum(rng.normal(0, 0.04, len(dates))))
    spx = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": close, "High": close, "Low": close,
                        "Close": close, "Adj Close": close, "Volume": 1})
    vix = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Open": vix_close, "High": vix_close, "Low": vix_close,
                        "Close": vix_close, "Adj Close": vix_close, "Volume": 0})
    if vix_extra_date:
        vix = pd.concat([vix, pd.DataFrame([{**vix.iloc[0].to_dict(), "Date": vix_extra_date, "Close": 99.0}])])
        vix = vix.sort_values("Date")
    series = {}
    for name, frame in (("gspc", spx), ("vix", vix)):
        text = frame.to_csv(index=False, lineterminator="\n")
        (raw / f"{name}.csv").write_text(text, encoding="utf-8", newline="\n")
        series[name] = {"file": f"{name}.csv", "sha256": hashlib.sha256(text.encode()).hexdigest()}
    meta = {"series": series, "last_completed_trading_date_in_all_series": "2005-12-30", "downloaded_at_utc": "synthetic"}
    (raw / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def run(tmp: Path, settings=SETTINGS, force=False) -> int:
    with redirect_stdout(io.StringIO()):
        return bmd.run(tmp / "raw", tmp / "processed", tmp / "reports", settings, force)


def hashes(directory: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())} if directory.exists() else {}


class TestBuild(unittest.TestCase):
    def test_outputs_counts_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            raw_before = hashes(tmp / "raw")
            self.assertEqual(run(tmp), 0)
            data = pd.read_csv(tmp / "processed" / "dataset.csv", float_precision="round_trip")
            meta = json.loads((tmp / "processed" / "metadata.json").read_text(encoding="utf-8"))
            n_spx = len(pd.bdate_range("1995-01-02", "2005-12-30"))
            self.assertEqual(len(data), n_spx - 20)                     # final 20 rows dropped
            self.assertEqual(meta["target"]["rows_dropped_incomplete_horizon"], 20)
            self.assertEqual(set(data["target"].unique()) <= {0, 1}, True)
            self.assertEqual(int(data["eligible"].sum()), meta["rows"]["eligible"])
            self.assertFalse(data.loc[data["eligible"], ["ret_1", "drawdown_252", "vix_pct_252"]].isna().any().any())
            self.assertEqual(meta["output"]["dataset.csv"], hashlib.sha256((tmp / "processed" / "dataset.csv").read_bytes()).hexdigest())
            self.assertEqual(hashes(tmp / "raw"), raw_before)
            self.assertIn("dataset_report.md", hashes(tmp / "reports"))
            self.assertEqual(len(meta["splits"]["folds"]), 5)
            self.assertIn("versions", meta)

    def test_vix_only_date_is_ignored_and_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw", vix_extra_date="1999-07-04")  # a Sunday: never an S&P 500 session
            self.assertEqual(run(tmp), 0)
            meta = json.loads((tmp / "processed" / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["vix_alignment"]["vix_only_dates_ignored"], ["1999-07-04"])
            data = pd.read_csv(tmp / "processed" / "dataset.csv")
            self.assertNotIn(99.0, data["vix_lag1"].to_numpy())

    def test_no_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            self.assertEqual(run(tmp), 0)
            before = (hashes(tmp / "processed"), hashes(tmp / "reports"))
            self.assertEqual(run(tmp), 1)
            self.assertEqual((hashes(tmp / "processed"), hashes(tmp / "reports")), before)
            self.assertEqual(run(tmp, force=True), 0)

    def test_failed_checks_write_nothing(self):
        cases = {
            "nan close": dict(spx_nan_at=500),
        }
        for name, kwargs in cases.items():
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                write_snapshot(tmp / "raw", **kwargs)
                self.assertEqual(run(tmp), 1, name)
                self.assertFalse((tmp / "processed").exists() or (tmp / "reports").exists(), name)

    def test_checksum_or_end_date_mismatch_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            with (tmp / "raw" / "vix.csv").open("a", encoding="utf-8") as f:
                f.write("\n")
            self.assertEqual(run(tmp), 1)
            self.assertFalse((tmp / "processed").exists())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            settings = copy.deepcopy(SETTINGS)
            settings["market"]["end_date"] = "2005-12-29"
            self.assertEqual(run(tmp, settings), 1)
            self.assertFalse((tmp / "processed").exists())

    def test_invalid_split_configuration_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_snapshot(tmp / "raw")
            settings = copy.deepcopy(SETTINGS)
            settings["evaluation"]["purge_gap_trading_days"] = 5
            self.assertEqual(run(tmp, settings), 1)
            self.assertFalse((tmp / "processed").exists())


if __name__ == "__main__":
    unittest.main()
