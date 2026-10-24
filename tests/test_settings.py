"""Checks on config/settings.yaml that guard the market-stress target definition.

Uses only the standard library (PyYAML is not installed yet), so values are
read with a simple key lookup. Every key read here is unique in the file.
"""

import re
import unittest
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def read_setting(key: str) -> str:
    text = SETTINGS_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\s#]+)", text, re.MULTILINE)
    if match is None:
        raise KeyError(f"{key} not found in {SETTINGS_PATH}")
    return match.group(1)


class TestMarketStressSettings(unittest.TestCase):
    def test_horizon_is_20_trading_days(self):
        self.assertEqual(int(read_setting("horizon_trading_days")), 20)

    def test_drawdown_threshold_is_5_percent(self):
        self.assertAlmostEqual(float(read_setting("drawdown_threshold")), 0.05)

    def test_purge_gap_covers_label_horizon(self):
        horizon = int(read_setting("horizon_trading_days"))
        purge_gap = int(read_setting("purge_gap_trading_days"))
        self.assertGreaterEqual(purge_gap, horizon)


if __name__ == "__main__":
    unittest.main()
