"""Download a frozen daily snapshot of ^GSPC and ^VIX from Yahoo Finance into data/raw/market/.

Usage:
    python scripts/download_market_data.py            # stop if a snapshot already exists
    python scripts/download_market_data.py --force    # replace an existing snapshot
    python scripts/download_market_data.py --output-dir DIR

The date range comes from config/settings.yaml (`market.start_date`,
`market.end_date`). When `end_date` is null, the request ends on the day before
today's New York date, so only completed trading days are included. Both
series are validated before anything is written; on any error nothing is saved.

Yahoo Finance access through yfinance is unofficial and intended for personal,
non-commercial use (docs/decisions.md, D6). Historical values can be revised,
so the snapshot's SHA-256 checksums are recorded in metadata.json.
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from fincrisis.data import market

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXCHANGE_TZ = ZoneInfo("America/New_York")


def load_settings() -> tuple[dict, dict]:
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    return settings["market"], settings["paths"]


def file_stem(symbol: str) -> str:
    return symbol.lstrip("^").lower()


def resolve_range(market_settings: dict, today_exchange: date) -> tuple[date, date, str]:
    start = date.fromisoformat(str(market_settings["start_date"]))
    if market_settings.get("end_date"):
        return start, date.fromisoformat(str(market_settings["end_date"])), "config market.end_date"
    return start, today_exchange - timedelta(days=1), "day before today's New York date (end_date is null)"


def fetch_daily(symbol: str, start: date, end_inclusive: date) -> pd.DataFrame:
    """Download unadjusted daily bars; yfinance's `end` is exclusive."""
    import yfinance as yf
    return yf.download(symbol, start=start.isoformat(), end=(end_inclusive + timedelta(days=1)).isoformat(),
                       interval="1d", auto_adjust=False, actions=False, progress=False, threads=False,
                       multi_level_index=False, timeout=30)


def package_versions() -> dict:
    versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
    for name in ("yfinance", "curl_cffi"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = None
    return versions


def write_outputs(outputs: dict[Path, str]) -> None:
    """Write all outputs to temporary files, then move them into place; clean up on error."""
    temps = {path: path.with_suffix(path.suffix + ".part") for path in outputs}
    try:
        for path, content in outputs.items():
            temps[path].write_text(content, encoding="utf-8", newline="\n")
        for path, tmp in temps.items():
            tmp.replace(path)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)


def run(out_dir: Path, symbols: dict[str, str], start: date, end_inclusive: date, range_source: str,
        force: bool, fetch=fetch_daily) -> int:
    paths = {name: out_dir / f"{file_stem(symbol)}.csv" for name, symbol in symbols.items()}
    meta_path = out_dir / "metadata.json"
    existing = [str(p) for p in [*paths.values(), meta_path] if p.exists()]
    if existing and not force:
        print(f"Snapshot file(s) already exist: {existing}\nNothing was downloaded or written. Use --force to replace.")
        return 1

    frames, summaries, errors = {}, {}, []
    for name, symbol in symbols.items():
        print(f"Downloading {symbol} {start} .. {end_inclusive}")
        try:
            frame = market.normalize_download(fetch(symbol, start, end_inclusive))
        except market.MarketDataError as error:
            errors.append(f"{symbol}: {error}")
            continue
        except Exception as error:  # network, rate limiting, or a changed Yahoo response
            errors.append(f"{symbol}: download failed ({type(error).__name__}: {error})")
            continue
        summary = market.validate_series(frame, symbol, start, end_inclusive)
        errors += [f"{symbol}: {e}" for e in summary["errors"]]
        frames[name], summaries[name] = frame, summary
    if errors:
        print("Validation failed; nothing was written:\n  " + "\n  ".join(errors))
        return 1

    names = list(symbols)
    calendar = market.compare_calendars(frames[names[0]], frames[names[1]], names[0], names[1])
    texts = {name: market.to_csv_text(frame) for name, frame in frames.items()}
    last_common = min(pd.Timestamp(s["last_date"]) for s in summaries.values()).date()
    metadata = {
        "source": "Yahoo Finance via yfinance (unofficial; personal, non-commercial use)",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested": {"tickers": symbols, "start": start.isoformat(), "end_inclusive": end_inclusive.isoformat(),
                      "end_source": range_source, "interval": "1d", "auto_adjust": False},
        "last_completed_trading_date_in_all_series": last_common.isoformat(),
        "series": {name: {**summaries[name], "file": paths[name].name,
                          "sha256": hashlib.sha256(texts[name].encode("utf-8")).hexdigest()} for name in names},
        "calendar_comparison": calendar,
        "warnings": [f"{symbols[n]}: {w}" for n in names for w in summaries[n]["warnings"]],
        "versions": package_versions(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs({**{paths[n]: texts[n] for n in names},
                   meta_path: json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"})
    for name in names:
        s = metadata["series"][name]
        print(f"Saved {paths[name]}: {s['rows']} rows, {s['first_date']} .. {s['last_date']}, sha256 {s['sha256']}")
    print(f"Last completed trading date in all series: {last_common}")
    print(f"Warnings: {len(metadata['warnings'])} (see {meta_path})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="replace an existing snapshot")
    parser.add_argument("--output-dir", type=Path, help="output directory (default: <raw_data>/market from config)")
    args = parser.parse_args()

    market_settings, paths = load_settings()
    symbols = {"gspc": market_settings["ticker"], "vix": market_settings["vix_ticker"]}
    start, end_inclusive, source = resolve_range(market_settings, datetime.now(EXCHANGE_TZ).date())
    out_dir = args.output_dir or PROJECT_ROOT / paths["raw_data"] / "market"
    return run(out_dir, symbols, start, end_inclusive, source, args.force)


if __name__ == "__main__":
    sys.exit(main())
