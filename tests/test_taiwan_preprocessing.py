"""Tests for Taiwanese Bankruptcy preprocessing (docs/decisions.md, D8).

Unit tests use small synthetic frames. End-to-end tests run the script on the
real raw CSV into a temporary directory and are skipped if the data is absent.
"""

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fincrisis.data import taiwan_bankruptcy as tb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "taiwan_bankruptcy"
SCRIPT = PROJECT_ROOT / "scripts" / "preprocess_taiwan_bankruptcy.py"
DROPPED = {" Current Liability to Liability", " Current Liability to Equity", " Net worth/Assets", " Net Income Flag"}


def synthetic(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    debt = rng.random(n)
    current_liab, current_equity = rng.random(n), rng.random(n)
    return pd.DataFrame({
        "Bankrupt?": (np.arange(n) % 10 == 0).astype("int64"),
        " Current Liabilities/Liability": current_liab,
        " Current Liability to Liability": current_liab.copy(),
        " Current Liabilities/Equity": current_equity,
        " Current Liability to Equity": current_equity.copy(),
        " Debt ratio %": debt,
        " Net worth/Assets": 1 - debt,
        " Net Income Flag": np.ones(n, dtype="int64"),
        " Liability-Assets Flag": (np.arange(n) == 3).astype("int64"),
        " Revenue Per Share (Yuan ¥)": rng.random(n) * 1e9,
    })


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestNameCleaning(unittest.TestCase):
    def test_examples(self):
        self.assertEqual(tb.clean_name("Bankrupt?"), "bankrupt")
        self.assertEqual(tb.clean_name(" Revenue Per Share (Yuan ¥)"), "revenue_per_share_yuan")
        self.assertEqual(tb.clean_name(" Debt ratio %"), "debt_ratio_pct")
        self.assertEqual(tb.clean_name(" Current Liabilities/Liability"), "current_liabilities_liability")
        self.assertEqual(tb.clean_name(" ROA(C) before interest and depreciation before interest"),
                         "roa_c_before_interest_and_depreciation_before_interest")

    def test_rejects_unhandled_non_ascii_and_empty(self):
        with self.assertRaises(tb.ValidationError):
            tb.clean_name(" Revenue (€)")
        with self.assertRaises(tb.ValidationError):
            tb.clean_name(" ?? ")

    def test_rejects_collisions(self):
        with self.assertRaises(tb.ValidationError):
            tb.clean_names([" Cash / Assets", "Cash-Assets"])
        with self.assertRaises(tb.ValidationError):
            tb.clean_names(["Row ID"])


class TestDropValidation(unittest.TestCase):
    def test_refuses_when_duplicate_differs(self):
        df = synthetic()
        df.loc[5, " Current Liability to Liability"] += 1e-9
        with self.assertRaisesRegex(tb.ValidationError, "no longer identical"):
            tb.validate_drops(df)

    def test_refuses_when_complement_fails(self):
        df = synthetic()
        df.loc[5, " Net worth/Assets"] += 1e-6
        with self.assertRaisesRegex(tb.ValidationError, "deviates from 1"):
            tb.validate_drops(df)

    def test_refuses_when_constant_varies(self):
        df = synthetic()
        df.loc[5, " Net Income Flag"] = 0
        with self.assertRaisesRegex(tb.ValidationError, "no longer constant"):
            tb.validate_drops(df)

    def test_refuses_when_column_missing(self):
        with self.assertRaisesRegex(tb.ValidationError, "not found"):
            tb.validate_drops(synthetic().drop(columns=" Debt ratio %"))

    def test_refuses_bad_target(self):
        df = synthetic()
        df.loc[0, "Bankrupt?"] = 2
        with self.assertRaises(tb.ValidationError):
            tb.preprocess(df)


class TestPreprocess(unittest.TestCase):
    def test_drops_exactly_the_four_columns(self):
        raw = synthetic()
        processed, column_map = tb.preprocess(raw)
        self.assertEqual(set(column_map.loc[column_map["status"] == "dropped", "original_name"]), DROPPED)
        self.assertEqual(list(processed.columns), [
            "row_id", "bankrupt", "current_liabilities_liability", "current_liabilities_equity",
            "debt_ratio_pct", "liability_assets_flag", "revenue_per_share_yuan",
        ])

    def test_values_and_input_unchanged(self):
        raw = synthetic()
        before = raw.copy()
        processed, column_map = tb.preprocess(raw)
        pd.testing.assert_frame_equal(raw, before)
        kept = column_map[column_map["status"] == "kept"]
        for original, clean in zip(kept["original_name"], kept["clean_name"]):
            self.assertTrue(np.array_equal(processed[clean].to_numpy(), raw[original].to_numpy()), clean)
        self.assertTrue(np.array_equal(processed["row_id"], np.arange(len(raw))))

    def test_column_map_covers_every_column_once(self):
        raw = synthetic()
        _, column_map = tb.preprocess(raw)
        self.assertEqual(list(column_map["original_name"]), list(raw.columns))
        self.assertEqual(list(column_map["original_position"]), list(range(raw.shape[1])))
        self.assertTrue(column_map["clean_name"].is_unique)


class TestSplit(unittest.TestCase):
    def test_disjoint_complete_stratified_deterministic(self):
        processed, _ = tb.preprocess(synthetic(200))
        split = tb.make_split(processed, test_size=0.2, seed=42)
        self.assertEqual(sorted(split["row_id"]), list(range(200)))
        self.assertEqual(set(split["split"]), {"train", "test"})
        test_ids = split.loc[split["split"] == "test", "row_id"]
        self.assertEqual(len(test_ids), 40)
        self.assertEqual(int(processed.loc[test_ids, "bankrupt"].sum()), 4)  # 20 positives * 0.2
        pd.testing.assert_frame_equal(split, tb.make_split(processed, test_size=0.2, seed=42))


class TestCsvRoundTrip(unittest.TestCase):
    def test_exact_round_trip(self):
        df = pd.DataFrame({"a": [0.1, 1 / 3, 6.4e9, 1e-300, 0.000182092597429571], "b": [1, 2, 3, 4, 5]})
        text = tb.to_csv_text(df)
        parsed = pd.read_csv(io.StringIO(text), float_precision="round_trip")
        pd.testing.assert_frame_equal(parsed, df, check_exact=True)


@unittest.skipUnless((RAW_DIR / "data.csv").exists(), "raw dataset not downloaded")
class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "out"
        cls.recorded = json.loads((RAW_DIR / "metadata.json").read_text(encoding="utf-8"))["csv_sha256"]
        cls.raw_before = sha256(RAW_DIR / "data.csv")
        cls.result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output-dir", str(cls.out)],
                                 capture_output=True, text=True)
        cls.raw = pd.read_csv(RAW_DIR / "data.csv", float_precision="round_trip")
        cls.data = pd.read_csv(cls.out / "data.csv", float_precision="round_trip")
        cls.column_map = pd.read_csv(cls.out / "column_map.csv", keep_default_na=False)
        cls.split = pd.read_csv(cls.out / "split.csv")
        cls.meta = json.loads((cls.out / "metadata.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_script_succeeded(self):
        self.assertEqual(self.result.returncode, 0, self.result.stdout + self.result.stderr)

    def test_raw_csv_unchanged(self):
        self.assertEqual(self.raw_before, self.recorded)
        self.assertEqual(sha256(RAW_DIR / "data.csv"), self.recorded)

    def test_shape_and_order(self):
        self.assertEqual(self.data.shape, (6819, 93))
        self.assertEqual(list(self.data.columns[:2]), ["row_id", "bankrupt"])
        self.assertTrue(np.array_equal(self.data["row_id"], np.arange(6819)))

    def test_kept_values_identical_to_raw(self):
        kept = self.column_map[self.column_map["status"] == "kept"]
        self.assertEqual(len(kept), 92)
        self.assertEqual(set(self.column_map.loc[self.column_map["status"] == "dropped", "original_name"]), DROPPED)
        for original, clean in zip(kept["original_name"], kept["clean_name"]):
            self.assertTrue(np.array_equal(self.data[clean].to_numpy(), self.raw[original].to_numpy()), clean)

    def test_target_unchanged(self):
        self.assertTrue(np.array_equal(self.data["bankrupt"].to_numpy(), self.raw["Bankrupt?"].to_numpy()))
        self.assertEqual(self.data["bankrupt"].value_counts().to_dict(), {0: 6599, 1: 220})
        self.assertIn("unverified", self.meta["target"]["label_meaning"])

    def test_split(self):
        self.assertEqual(sorted(self.split["row_id"]), list(range(6819)))
        target = self.data.set_index("row_id")["bankrupt"]
        for name, size in [("test", 1364), ("train", 5455)]:
            ids = self.split.loc[self.split["split"] == name, "row_id"]
            self.assertEqual(len(ids), size)
            self.assertAlmostEqual(target.loc[ids].mean(), 220 / 6819, delta=0.001)

    def test_metadata_checksums_match_outputs(self):
        for name, digest in self.meta["output"]["sha256"].items():
            self.assertEqual(sha256(self.out / name), digest, name)
        self.assertEqual(self.meta["input"]["sha256"], self.recorded)

    def test_no_overwrite_without_force(self):
        before = {p.name: sha256(p) for p in self.out.iterdir()}
        rerun = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output-dir", str(self.out)],
                               capture_output=True, text=True)
        self.assertEqual(rerun.returncode, 1)
        self.assertEqual({p.name: sha256(p) for p in self.out.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
