"""Inspect the raw Taiwanese Bankruptcy CSV and write a data-quality report.

Usage:
    python scripts/inspect_taiwan_bankruptcy.py           # stop if reports already exist
    python scripts/inspect_taiwan_bankruptcy.py --force   # overwrite existing reports

Reads data/raw/taiwan_bankruptcy/data.csv without modifying it, verifies its
SHA-256 against metadata.json before and after analysis, and writes
inspection_report.md and feature_summary.csv to data/reports/taiwan_bankruptcy/.

Observations only: no cleaning, no modelling. The meaning of target labels
0 and 1 is not documented by UCI and is not inferred here.
"""

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

TARGET = "Bankrupt?"
NEAR_CONSTANT_SHARE = 0.99  # one value covers at least this share of rows
LARGE_VALUE = 1e6           # absolute magnitude flagged for investigation
HIGH_CORRELATION = 0.95     # |r| flagged as near-duplicate information

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_paths() -> tuple[Path, Path]:
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    paths = yaml.safe_load(settings_path.read_text(encoding="utf-8"))["paths"]
    return (
        PROJECT_ROOT / paths["raw_data"] / "taiwan_bankruptcy",
        PROJECT_ROOT / paths["data_reports"] / "taiwan_bankruptcy",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(path: Path) -> dict | None:
    """Return metadata.json as a dict, or None if missing or invalid."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(metadata, dict) or not isinstance(metadata.get("csv_sha256"), str):
        return None
    return metadata


def fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return "n/a" if np.isnan(value) else f"{value:.6g}"
    return str(value)


def md_table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "None found."
    cell = lambda v: fmt(v).replace("|", "\\|")
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)


def build_feature_summary(numeric: pd.DataFrame, raw_names: dict[str, str]) -> pd.DataFrame:
    stats = numeric.describe().T.drop(columns="count")
    summary = pd.DataFrame({
        "raw_name": [raw_names[c] for c in numeric.columns],
        "dtype": numeric.dtypes.astype(str),
        "count": numeric.count(),
        "missing": numeric.isna().sum(),
        "infinite": np.isinf(numeric).sum(),
    }, index=numeric.columns)
    summary = summary.join(stats)
    summary["n_unique"] = numeric.nunique(dropna=False)
    summary["zero_share"] = (numeric == 0).mean()
    summary["top_value_share"] = numeric.apply(lambda s: s.value_counts(dropna=False).iloc[0] / len(s))
    summary["n_negative"] = (numeric < 0).sum()
    summary["n_above_1"] = (numeric > 1).sum()
    summary["n_abs_above_large"] = (numeric.abs() > LARGE_VALUE).sum()
    summary.index.name = "feature"
    return summary


def analyze(df: pd.DataFrame, header: list[str], context: dict) -> tuple[str, pd.DataFrame]:
    sections, issues = [], []
    n_rows = len(df)

    # 1. Source
    sections.append("## 1. Source\n\n" + md_table(["Item", "Value"], [[k, v] for k, v in context.items()]))

    # 2. Dimensions and column names (header read directly; pandas renames duplicates)
    stripped = [name.strip() for name in header]
    padded = [name for name in header if name != name.strip()]
    dup_raw = sorted({n for n in header if header.count(n) > 1})
    dup_stripped = sorted({n for n in stripped if stripped.count(n) > 1})
    non_ascii = [
        [repr(name), ", ".join(f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}" for ch in name if ord(ch) > 127)]
        for name in header if any(ord(ch) > 127 for ch in name)
    ]
    text = (
        f"## 2. Dimensions and column names\n\n"
        f"- Rows: {n_rows}\n- Columns: {df.shape[1]} (1 target + {df.shape[1] - 1} features)\n"
        f"- Names with leading/trailing whitespace: {len(padded)}\n"
        f"- Duplicate names as stored: {dup_raw or 'none'}\n"
        f"- Duplicate names after stripping whitespace: {dup_stripped or 'none'}\n\n"
        f"### Names containing non-ASCII characters\n\n" + md_table(["Name (as stored)", "Characters"], non_ascii)
        + "\n\n### All columns (as stored)\n\n"
        + md_table(["#", "Name"], [[i, repr(name)] for i, name in enumerate(header)])
    )
    sections.append(text)
    if padded:
        issues.append(f"{len(padded)} column names have leading/trailing whitespace; strip them in a cleaning step.")
    if dup_raw or dup_stripped:
        issues.append(f"Duplicate column names: {dup_raw or dup_stripped}.")
    if non_ascii:
        issues.append(f"{len(non_ascii)} column name(s) contain non-ASCII characters; consider ASCII-safe names.")

    names = dict(zip(df.columns, stripped))
    if not dup_stripped:
        df = df.rename(columns=names)
    raw_names = {stripped_name: raw for stripped_name, raw in zip(df.columns, header)}

    if TARGET not in df.columns:
        issues.append(f"Target column {TARGET!r} not found.")
        features = df
    else:
        features = df.drop(columns=TARGET)
    numeric = features.select_dtypes("number")
    non_numeric = [c for c in features.columns if c not in numeric.columns]

    # 3. Data types, missing and infinite values
    dtype_counts = df.dtypes.astype(str).value_counts()
    missing = df.isna().sum()
    infinite = np.isinf(numeric).sum()
    sections.append(
        "## 3. Data types, missing and infinite values\n\n"
        + md_table(["dtype", "Columns"], [[k, v] for k, v in dtype_counts.items()])
        + "\n\nColumns not stored as float64:\n\n"
        + md_table(["Column", "dtype"], [[c, str(t)] for c, t in df.dtypes.items() if str(t) != "float64"])
        + f"\n\n- Non-numeric feature columns: {non_numeric or 'none'}"
        + f"\n- Missing values: {int(missing.sum())} in total, in {int((missing > 0).sum())} column(s)"
        + f"\n- Infinite values: {int(infinite.sum())} in total, in {int((infinite > 0).sum())} column(s)\n\n"
        + md_table(["Column", "Missing"], [[c, int(v)] for c, v in missing.items() if v > 0])
    )
    if non_numeric:
        issues.append(f"Non-numeric feature columns: {non_numeric}.")
    if missing.sum():
        issues.append(f"{int(missing.sum())} missing values; decide on handling before training.")
    if infinite.sum():
        issues.append(f"{int(infinite.sum())} infinite values; investigate their origin.")

    # 4. Duplicates
    exact_dups = int(df.duplicated().sum())
    feature_dup_mask = features.duplicated(keep=False)
    conflicting = 0
    if TARGET in df.columns and feature_dup_mask.any():
        groups = df[feature_dup_mask].groupby(list(features.columns), dropna=False)[TARGET].nunique()
        conflicting = int((groups > 1).sum())
    identical = defaultdict(list)
    for col in numeric.columns:
        identical[numeric[col].to_numpy(dtype=float).tobytes()].append(col)
    dup_columns = [cols for cols in identical.values() if len(cols) > 1]
    sections.append(
        "## 4. Duplicates\n\n"
        f"- Exact duplicate rows (beyond first occurrence): {exact_dups}\n"
        f"- Rows sharing identical feature values with another row: {int(feature_dup_mask.sum())}\n"
        f"- Groups of identical features with different target values: {conflicting}\n\n"
        "Columns with identical values under different names:\n\n"
        + md_table(["Columns"], [[", ".join(cols)] for cols in dup_columns])
    )
    if exact_dups:
        issues.append(f"{exact_dups} exact duplicate rows; decide whether to drop before splitting.")
    if conflicting:
        issues.append(f"{conflicting} groups of identical features carry different target values.")
    if dup_columns:
        issues.append(f"{len(dup_columns)} group(s) of identical columns; keep one of each.")

    # 5. Target
    if TARGET in df.columns:
        counts = df[TARGET].value_counts(dropna=False).sort_index()
        values = set(counts.index.tolist())
        sections.append(
            f"## 5. Target column `{TARGET}`\n\n"
            + md_table(["Value", "Count", "Proportion"], [[v, int(c), c / n_rows] for v, c in counts.items()])
            + "\n\n**The meanings of labels 0 and 1 are unverified.** UCI does not document them, "
            "and they are not inferred from class counts here."
        )
        if values != {0, 1}:
            issues.append(f"Target values are {sorted(values)}, expected exactly 0 and 1.")
        else:
            minority = counts.min() / n_rows
            issues.append(
                f"Class imbalance: the smaller class is {minority:.2%} of rows; use stratified splits and "
                "imbalance-aware metrics (e.g. PR-AUC)."
            )
        issues.append("Label meanings (0/1) are not documented by UCI; confirm from an authoritative source.")

    # 6. Numerical summaries
    summary = build_feature_summary(numeric, raw_names)
    sections.append(
        "## 6. Numerical feature summaries\n\n"
        "Full statistics are in `feature_summary.csv`.\n\n"
        + md_table(
            ["Feature", "min", "median", "mean", "max", "std", "n_unique"],
            [[f, r["min"], r["50%"], r["mean"], r["max"], r["std"], int(r["n_unique"])] for f, r in summary.iterrows()],
        )
    )

    # 7. Constant, near-constant, and two-valued features
    constant = summary.index[summary["n_unique"] == 1].tolist()
    near_constant = summary.index[(summary["n_unique"] > 1) & (summary["top_value_share"] >= NEAR_CONSTANT_SHARE)]
    two_valued = summary.index[summary["n_unique"] == 2].tolist()
    sections.append(
        "## 7. Constant, near-constant, and two-valued features\n\n"
        f"- Constant: {constant or 'none'}\n"
        f"- Near-constant (one value in >= {NEAR_CONSTANT_SHARE:.0%} of rows):\n\n"
        + md_table(["Feature", "Top value share", "n_unique"],
                   [[f, summary.at[f, "top_value_share"], int(summary.at[f, "n_unique"])] for f in near_constant])
        + "\n\nTwo-valued features:\n\n"
        + md_table(["Feature", "Values"], [[f, sorted(numeric[f].unique().tolist())] for f in two_valued])
    )
    if constant:
        issues.append(f"Constant feature(s) carry no information: {constant}.")
    if len(near_constant):
        issues.append(f"{len(near_constant)} near-constant feature(s); check whether they are informative.")

    # 8. Suspicious values
    negative = summary.index[summary["n_negative"] > 0].tolist()
    above_1 = summary[summary["n_above_1"] > 0].sort_values("n_above_1", ascending=False)
    sections.append(
        "## 8. Suspicious values\n\n"
        "These are observations requiring investigation, not confirmed errors. Extreme values may be "
        "genuine, differently scaled, or placeholder codes.\n\n"
        f"- Features with negative values: {negative or 'none'}\n\n"
        f"Features with values above 1 (|value| > {LARGE_VALUE:g} counted separately):\n\n"
        + md_table(
            ["Feature", "Rows > 1", "Share > 1", f"Rows abs > {LARGE_VALUE:g}", "Rows <= 1", "median", "max"],
            [[f, int(r["n_above_1"]), r["n_above_1"] / n_rows, int(r["n_abs_above_large"]),
              n_rows - int(r["n_above_1"]), r["50%"], r["max"]] for f, r in above_1.iterrows()],
        )
    )
    large = above_1[above_1["n_abs_above_large"] > 0]
    if len(large):
        issues.append(
            f"{len(large)} feature(s) contain values with |value| > {LARGE_VALUE:g}; investigate whether they are "
            "genuine, differently scaled, or placeholders before choosing a treatment."
        )

    # 9. Scale comparison
    in_unit = summary.index[(summary["min"] >= 0) & (summary["max"] <= 1)]
    outside = summary.drop(index=in_unit)
    magnitudes = np.floor(np.log10(outside["max"].abs().where(outside["max"] != 0))).value_counts().sort_index()
    majority_above_1 = outside.index[outside["n_above_1"] > n_rows / 2].tolist()
    sections.append(
        "## 9. Scale comparison\n\n"
        f"- Features with all values in [0, 1]: {len(in_unit)}\n"
        f"- Features with values outside [0, 1]: {len(outside)}\n"
        f"- Of those, features where more than half of rows exceed 1: {majority_above_1 or 'none'}\n\n"
        "Order of magnitude of the maximum, for features outside [0, 1]:\n\n"
        + md_table(["10^k", "Features"], [[int(k), int(v)] for k, v in magnitudes.items()])
        + "\n\nThis is an observation: features do not share a common scale, so the dataset should not be "
        "assumed to be uniformly normalized."
    )
    if len(outside) and len(in_unit):
        issues.append(
            f"Mixed scales: {len(in_unit)} features lie in [0, 1] while {len(outside)} exceed it; "
            "record this before relying on any 'already normalized' assumption."
        )

    # 10. Highly correlated pairs
    usable = numeric.drop(columns=constant)
    pearson, spearman = usable.corr(method="pearson"), usable.corr(method="spearman")
    pairs = []
    cols = usable.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            p, s = pearson.at[a, b], spearman.at[a, b]
            if max(abs(p), abs(s)) >= HIGH_CORRELATION:
                pairs.append([a, b, p, s])
    pairs.sort(key=lambda row: -max(abs(row[2]), abs(row[3])))
    sections.append(
        f"## 10. Highly correlated pairs (|Pearson| or |Spearman| >= {HIGH_CORRELATION})\n\n"
        "Constant features excluded. Pearson is sensitive to extreme values; Spearman uses ranks.\n\n"
        + md_table(["Feature A", "Feature B", "Pearson", "Spearman"], pairs)
    )
    if pairs:
        issues.append(f"{len(pairs)} highly correlated feature pair(s); consider redundancy before training.")

    # 11. Issues
    sections.append(
        "## 11. Potential issues to investigate before training\n\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nThis list describes observations only; treatments are separate decisions."
    )

    report = "# Taiwanese Bankruptcy Dataset: Inspection Report\n\n" + "\n\n".join(sections) + "\n"
    return report, summary


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing reports")
    args = parser.parse_args()

    raw_dir, report_dir = load_paths()
    csv_path, meta_path = raw_dir / "data.csv", raw_dir / "metadata.json"
    report_path, summary_path = report_dir / "inspection_report.md", report_dir / "feature_summary.csv"

    existing = [p for p in (report_path, summary_path) if p.exists()]
    if existing and not args.force:
        print(f"Report(s) already exist: {[str(p) for p in existing]}")
        print("Nothing was written. Use --force to overwrite.")
        return 1

    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}. Run scripts/download_taiwan_bankruptcy.py first.")
        return 1
    metadata = load_metadata(meta_path)
    if metadata is None:
        print(f"{meta_path} is missing or invalid; cannot verify the dataset. Nothing was written.")
        return 1
    checksum_before = file_sha256(csv_path)
    if checksum_before != metadata["csv_sha256"]:
        print(f"Checksum mismatch for {csv_path}; expected {metadata['csv_sha256']}. Nothing was written.")
        return 1

    with csv_path.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    df = pd.read_csv(csv_path, encoding="utf-8")

    context = {
        "Dataset": csv_path.relative_to(PROJECT_ROOT).as_posix(),
        "CSV SHA-256 (verified against metadata.json)": checksum_before,
        "Source URL": metadata.get("source_url", "n/a"),
        "DOI": metadata.get("doi", "n/a"),
        "License": metadata.get("license", "n/a"),
        "Retrieved (UTC)": metadata.get("retrieved_at_utc", "n/a"),
        "Report generated (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pandas / numpy": f"{pd.__version__} / {np.__version__}",
        "Thresholds": f"near-constant >= {NEAR_CONSTANT_SHARE}, large |value| > {LARGE_VALUE:g}, "
                      f"high correlation >= {HIGH_CORRELATION}",
    }
    report, summary = analyze(df, header, context)

    checksum_after = file_sha256(csv_path)
    if checksum_after != checksum_before:
        print(f"{csv_path} changed during analysis. Nothing was written.")
        return 1

    report_dir.mkdir(parents=True, exist_ok=True)
    write_outputs({report_path: report, summary_path: summary.to_csv()})
    print(f"Saved {report_path}")
    print(f"Saved {summary_path}")
    print("Raw CSV checksum unchanged after analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
