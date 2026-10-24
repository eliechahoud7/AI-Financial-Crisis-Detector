"""Build the market-stress dataset from the frozen ^GSPC and ^VIX snapshots (docs/decisions.md, D2, D11, D12).

Usage:
    python scripts/build_market_dataset.py            # stop if outputs exist
    python scripts/build_market_dataset.py --force    # overwrite existing outputs

Reads data/raw/market/ (never modified; checksums verified against its
metadata.json before and after) and writes:
    data/processed/market_stress/dataset.csv       features, target, eligibility
    data/processed/market_stress/metadata.json     checksums, config, counts, splits
    data/reports/market_stress/dataset_report.md   human-readable summary

No model is trained and nothing is fitted. The target uses future closes by
design; features use only information available before the close on day t
(S&P 500) or from the previous S&P 500 session (VIX).
"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fincrisis.evaluation.splits import describe_splits, walk_forward_splits
from fincrisis.features import market_features as mf
from fincrisis.features.market_target import stress_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ["Date", "Close"]


class BuildError(Exception):
    """An input or consistency check failed; nothing should be written."""


def load_settings() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_snapshot(raw_dir: Path, end_date: str | None) -> tuple[dict, dict]:
    """Return (raw metadata, verified checksums); raise if any file differs from metadata.json."""
    try:
        meta = json.loads((raw_dir / "metadata.json").read_text(encoding="utf-8"))
        series = meta["series"]
    except (OSError, ValueError, KeyError) as error:
        raise BuildError(f"Cannot read snapshot metadata in {raw_dir}: {error}") from error
    checksums = {}
    for name in ("gspc", "vix"):
        if name not in series:
            raise BuildError(f"Snapshot metadata has no '{name}' series")
        path = raw_dir / series[name]["file"]
        if not path.exists():
            raise BuildError(f"Snapshot file missing: {path}")
        actual = sha256_file(path)
        if actual != series[name]["sha256"]:
            raise BuildError(f"Checksum mismatch for {path.name}: {actual} != {series[name]['sha256']}")
        checksums[path.name] = actual
    if end_date and meta.get("last_completed_trading_date_in_all_series") != str(end_date):
        raise BuildError(f"Config market.end_date {end_date} does not match the snapshot's last common date "
                         f"{meta.get('last_completed_trading_date_in_all_series')}")
    return meta, checksums


def read_close(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path, float_precision="round_trip")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise BuildError(f"{path.name}: missing columns {missing}")
    dates = pd.to_datetime(frame["Date"], format="%Y-%m-%d", errors="coerce")
    if dates.isna().any():
        raise BuildError(f"{path.name}: {int(dates.isna().sum())} unparseable dates")
    close = pd.Series(frame["Close"].to_numpy(dtype=float), index=pd.DatetimeIndex(dates), name=name)
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise BuildError(f"{path.name}: dates must be unique and sorted")
    return close


def episodes(target: pd.Series) -> int:
    """Number of separate runs of consecutive label-1 rows."""
    values = target.to_numpy()
    return int(((values == 1) & (np.concatenate([[0], values[:-1]]) != 1)).sum())


def versions() -> dict:
    import sklearn
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__}


def write_outputs(outputs: dict[Path, str]) -> None:
    """Write all outputs to temporary files, then move them into place; clean up on error."""
    temps = {path: path.with_suffix(path.suffix + ".part") for path in outputs}
    try:
        for path, content in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temps[path].write_text(content, encoding="utf-8", newline="\n")
        for path, tmp in temps.items():
            tmp.replace(path)
    finally:
        for tmp in temps.values():
            tmp.unlink(missing_ok=True)


def build(raw_dir: Path, settings: dict) -> tuple[pd.DataFrame, dict, str]:
    """Return (dataset, metadata, report text). Raises BuildError on any failed check."""
    target_cfg, ms = settings["market_stress_target"], settings["market_stress"]
    horizon, threshold = target_cfg["horizon_trading_days"], target_cfg["drawdown_threshold"]
    purge = settings["evaluation"]["purge_gap_trading_days"]
    if target_cfg.get("peak_definition") != "forward_running_max":
        raise BuildError(f"Unsupported peak_definition {target_cfg.get('peak_definition')!r}")

    raw_meta, checksums_before = verify_snapshot(raw_dir, settings["market"].get("end_date"))
    spx = read_close(raw_dir / raw_meta["series"]["gspc"]["file"], "spx_close")
    vix = read_close(raw_dir / raw_meta["series"]["vix"]["file"], "vix_close")
    try:
        features, alignment = mf.build_features(spx, vix)
        target = stress_label(spx, horizon, threshold)
    except ValueError as error:
        raise BuildError(str(error)) from error

    dataset = features.copy()
    dataset.insert(0, "position", np.arange(len(dataset)))
    dataset["target"] = target
    dropped_tail = dataset.index[target.isna().to_numpy()]
    if len(dropped_tail) != horizon or not (dropped_tail == dataset.index[-horizon:]).all():
        raise BuildError(f"Expected exactly the last {horizon} rows to lack a target; found {len(dropped_tail)}")
    dataset = dataset.iloc[:-horizon].copy()
    dataset["target"] = dataset["target"].astype(int)
    missing_mask = dataset[mf.FEATURES].isna()
    dataset["eligible"] = ~missing_mask.any(axis=1)
    dataset["missing_features"] = [";".join(c for c, m in zip(mf.FEATURES, row) if m) for row in missing_mask.to_numpy()]

    years = list(range(ms["validation_first_year"], ms["validation_last_year"] + 1))
    try:
        splits = walk_forward_splits(dataset.index, dataset["eligible"].to_numpy(), years, ms["test_start"], purge, horizon)
    except ValueError as error:
        raise BuildError(f"Split generation failed: {error}") from error
    split_summary = describe_splits(splits, dataset.index, dataset["target"].to_numpy())

    if verify_snapshot(raw_dir, settings["market"].get("end_date"))[1] != checksums_before:
        raise BuildError("Raw snapshot changed during the build")

    eligible = dataset[dataset["eligible"]]
    first_eligible = eligible.index[0]
    ineligible = dataset[~dataset["eligible"]]
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {"directory": raw_dir.relative_to(PROJECT_ROOT).as_posix() if raw_dir.is_relative_to(PROJECT_ROOT) else str(raw_dir),
                  "sha256_verified_before_and_after": checksums_before,
                  "snapshot_downloaded_at_utc": raw_meta.get("downloaded_at_utc"),
                  "spx_rows": int(len(spx)), "vix_rows": int(len(vix)),
                  "spx_first_date": str(spx.index[0].date()), "spx_last_date": str(spx.index[-1].date())},
        "target": {"definition": "DD_t = min_{j=t+1..t+H} C_j / max(C_t..C_j) - 1; y_t = 1 if DD_t <= -threshold",
                   "horizon": horizon, "threshold": threshold, "tolerance": 1e-12,
                   "rows_dropped_incomplete_horizon": int(horizon),
                   "dropped_dates": [str(dropped_tail[0].date()), str(dropped_tail[-1].date())]},
        "features": mf.FEATURE_DEFINITIONS,
        "vix_alignment": {"rule": "D11: previous S&P 500 session's VIX close; VIX-only dates ignored; no filling",
                          **alignment},
        "rows": {"dataset": int(len(dataset)), "first_date": str(dataset.index[0].date()),
                 "last_date": str(dataset.index[-1].date()), "eligible": int(len(eligible)),
                 "ineligible": int(len(ineligible)), "first_eligible_date": str(first_eligible.date()),
                 "ineligible_after_first_eligible": int((~dataset.loc[first_eligible:, "eligible"]).sum())},
        "target_counts": {
            "all_rows": {"label_0": int((dataset.target == 0).sum()), "label_1": int((dataset.target == 1).sum()),
                         "prevalence": round(float(dataset.target.mean()), 6), "episodes": episodes(dataset.target)},
            "eligible_rows": {"label_0": int((eligible.target == 0).sum()), "label_1": int((eligible.target == 1).sum()),
                              "prevalence": round(float(eligible.target.mean()), 6), "episodes": episodes(eligible.target)},
        },
        "missingness": mf.missingness(dataset[mf.FEATURES]),
        "splits": {"validation_years": years, "test_start": ms["test_start"], **split_summary},
        "settings": {"market": settings["market"], "market_stress_target": target_cfg,
                     "evaluation": settings["evaluation"], "market_stress": ms},
        "versions": versions(),
    }
    dataset.index = dataset.index.strftime("%Y-%m-%d")
    dataset.index.name = "date"
    return dataset, meta, build_report(meta)


def build_report(meta: dict) -> str:
    rows, counts = meta["rows"], meta["target_counts"]
    s = meta["splits"]
    fold_lines = [f"| {f['validation_year']} | {f['train']['rows']} ({f['train']['first_date']} .. {f['train']['last_date']}) | "
                  f"{f['purged']['rows']} | {f['validation']['rows']} ({f['validation']['first_date']} .. "
                  f"{f['validation']['last_date']}) | {f['validation']['label_1']} | {f['validation']['prevalence']:.3f} |"
                  for f in s["folds"]]
    miss_lines = [f"| `{name}` | {m['missing']} | {m['warm_up']} | {m['after_warm_up']} |" for name, m in meta["missingness"].items()]
    feat_lines = [f"| `{name}` | {definition} |" for name, definition in meta["features"].items()]
    va = meta["vix_alignment"]

    def part(p):
        return f"{p['rows']} rows, {p.get('first_date', '-')} .. {p.get('last_date', '-')}, label 1: {p.get('label_1', 0)} ({p.get('prevalence', 0):.3f})"
    return "\n".join([
        "# Market-Stress Dataset Report",
        "",
        f"Generated (UTC): {meta['generated_at_utc']}. No model was trained; this report describes data only.",
        "",
        "## Input",
        "",
        f"- Snapshot `{meta['input']['directory']}` downloaded {meta['input']['snapshot_downloaded_at_utc']}; "
        f"checksums verified before and after the build: "
        + ", ".join(f"`{k}` `{v}`" for k, v in meta["input"]["sha256_verified_before_and_after"].items()),
        f"- S&P 500 rows: {meta['input']['spx_rows']} ({meta['input']['spx_first_date']} .. {meta['input']['spx_last_date']}); VIX rows: {meta['input']['vix_rows']}",
        "",
        "## Target (D2)",
        "",
        f"`{meta['target']['definition']}`, H = {meta['target']['horizon']}, threshold = {meta['target']['threshold']} "
        "(a decline exactly at the threshold counts).",
        f"The last {meta['target']['rows_dropped_incomplete_horizon']} rows ({meta['target']['dropped_dates'][0]} .. "
        f"{meta['target']['dropped_dates'][1]}) have no complete future window and were dropped, not labelled 0.",
        "",
        "| Rows | Label 0 | Label 1 | Prevalence | Separate label-1 episodes |",
        "|---|---|---|---|---|",
        *[f"| {name.replace('_', ' ')} | {c['label_0']} | {c['label_1']} | {c['prevalence']:.4f} | {c['episodes']} |"
          for name, c in counts.items()],
        "",
        "Consecutive label-1 rows overlap in their future windows, so the number of episodes is a better "
        "sense of the effective sample size than the row count.",
        "",
        "## Rows",
        "",
        f"- Dataset: {rows['dataset']} rows, {rows['first_date']} .. {rows['last_date']}.",
        f"- Eligible for modelling (no missing feature): {rows['eligible']}; first eligible date {rows['first_eligible_date']}.",
        f"- Ineligible: {rows['ineligible']} (feature warm-up; listed per row in `missing_features`). "
        f"Ineligible rows after the first eligible date: {rows['ineligible_after_first_eligible']}.",
        "",
        "## Features",
        "",
        "S&P 500 features use closes up to and including day t; VIX features use the previous S&P 500 session (D11). "
        "Every rolling window needs a full window of values.",
        "",
        "| Feature | Definition |",
        "|---|---|",
        *feat_lines,
        "",
        "## Missing values",
        "",
        "| Feature | Missing | Warm-up (leading) | After warm-up |",
        "|---|---|---|---|",
        *miss_lines,
        "",
        "## VIX alignment (D11)",
        "",
        f"- VIX-only dates ignored (S&P 500 closed): {va['vix_only_dates_ignored'] or 'none'}",
        f"- S&P 500 dates without a VIX close: {va['spx_dates_without_vix_close'] or 'none'}",
        f"- Missing `vix_lag1` values: {va['vix_lag1_missing']} ({va['vix_lag1_missing_reason']})",
        "",
        "## Chronological splits (D12, for Phase C)",
        "",
        f"Purge gap: {s['purge']} sessions (horizon {s['horizon']}). Training rows end {s['purge']} sessions before each "
        "validation year; development rows end that many sessions before the test start.",
        "",
        "| Validation year | Training rows | Purged | Validation rows | Validation label 1 | Prevalence |",
        "|---|---|---|---|---|---|",
        *fold_lines,
        "",
        f"- Development (all folds' data): {part(s['development'])}",
        f"- Purged before test: {part(s['test_purged'])}",
        f"- Held-out test: {part(s['test'])}",
        "",
        "## Limitations",
        "",
        "- Yahoo Finance data is unofficial; values may differ from other vendors and may be revised.",
        "- Label-1 rows cluster in a small number of episodes; some validation years may contain few or none.",
        "- Nothing here is a forecast or a model result.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()
    settings = load_settings()
    paths = settings["paths"]
    return run(PROJECT_ROOT / paths["raw_data"] / "market",
               PROJECT_ROOT / paths["processed_data"] / "market_stress",
               PROJECT_ROOT / paths["data_reports"] / "market_stress", settings, args.force)


def run(raw_dir: Path, processed_dir: Path, report_dir: Path, settings: dict, force: bool) -> int:
    outputs = {"dataset": processed_dir / "dataset.csv", "metadata": processed_dir / "metadata.json",
               "report": report_dir / "dataset_report.md"}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing and not force:
        print(f"Output(s) already exist: {existing}\nNothing was written. Use --force to overwrite.")
        return 1
    try:
        dataset, meta, report = build(raw_dir, settings)
    except BuildError as error:
        print(f"Build failed: {error}\nNothing was written.")
        return 1
    csv_text = dataset.to_csv(lineterminator="\n")
    meta["output"] = {"dataset.csv": hashlib.sha256(csv_text.encode("utf-8")).hexdigest(),
                      "columns": ["date", *dataset.columns]}
    write_outputs({outputs["dataset"]: csv_text,
                   outputs["metadata"]: json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                   outputs["report"]: report})
    r, c = meta["rows"], meta["target_counts"]["eligible_rows"]
    print(f"Saved {outputs['dataset']} ({r['dataset']} rows, {r['first_date']} .. {r['last_date']}; {r['eligible']} eligible)")
    print(f"Eligible label 1: {c['label_1']} of {r['eligible']} ({c['prevalence']:.4f}), {c['episodes']} episodes")
    print(f"Saved {outputs['metadata']} and {outputs['report']}. Raw snapshot checksums unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
