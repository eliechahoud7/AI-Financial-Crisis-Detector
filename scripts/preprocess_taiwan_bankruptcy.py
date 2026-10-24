"""Preprocess the raw Taiwanese Bankruptcy CSV into data/interim/.

Usage:
    python scripts/preprocess_taiwan_bankruptcy.py            # stop if outputs exist
    python scripts/preprocess_taiwan_bankruptcy.py --force    # overwrite existing outputs
    python scripts/preprocess_taiwan_bankruptcy.py --output-dir DIR

Deterministic, target-independent steps only (docs/decisions.md, D8):
exact parsing, snake_case names with a mapping to the originals, removal of
four verified redundant or constant columns, and a stratified train/test split.
No values are changed and nothing is fitted. The raw CSV is verified against
metadata.json before and after processing and is never modified.

The meanings of target labels 0 and 1 are not documented by UCI and remain
unverified; this script does not assume which class represents bankruptcy.
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
import sklearn
import yaml

from fincrisis.data import taiwan_bankruptcy as tb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_CAVEAT = (
    "The meanings of target labels 0 and 1 are not documented by UCI and remain unverified. "
    "No assumption is made about which class represents bankruptcy."
)
OUTPUT_FILES = ["data.csv", "column_map.csv", "split.csv", "metadata.json", "preprocessing_report.md"]


def load_settings() -> dict:
    return yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def recorded_checksum(meta_path: Path) -> str | None:
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))["csv_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) else None


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


def large_value_features(processed: pd.DataFrame, column_map: pd.DataFrame) -> list[dict]:
    """Describe features with values above 1 (left unchanged); descriptive only."""
    originals = dict(zip(column_map["clean_name"], column_map["original_name"]))
    rows = []
    for col in processed.columns.drop([tb.ROW_ID, tb.TARGET]):
        above_1 = int((processed[col] > 1).sum())
        if above_1:
            rows.append({"clean_name": col, "original_name": originals[col],
                         "n_above_1": above_1, "n_above_1e6": int((processed[col] > 1e6).sum())})
    return sorted(rows, key=lambda r: -r["n_above_1"])


def class_counts(target: pd.Series) -> dict:
    counts = target.value_counts().sort_index()
    return {str(k): {"count": int(v), "share": round(v / len(target), 6)} for k, v in counts.items()}


def build_report(meta: dict, column_map: pd.DataFrame, large: list[dict]) -> str:
    dropped = column_map[column_map["status"] == "dropped"]
    lines = [
        "# Taiwanese Bankruptcy Dataset: Preprocessing Report",
        "",
        f"Generated (UTC): {meta['generated_at_utc']}",
        "",
        "## Input",
        "",
        f"- File: `{meta['input']['path']}`",
        f"- SHA-256: `{meta['input']['sha256']}` (matches metadata.json; unchanged after processing)",
        f"- Shape: {meta['input']['rows']} rows x {meta['input']['columns']} columns",
        "- Parsing: `pandas.read_csv(..., float_precision=\"round_trip\")` (exact decimal parsing)",
        "",
        "## Transformations",
        "",
        "1. Column names converted to ASCII snake_case (`¥` removed, `%` -> `pct`); "
        "target `Bankrupt?` renamed to `bankrupt`. No name collisions. Full mapping in `column_map.csv`.",
        f"2. Dropped {len(dropped)} columns after re-verifying each relationship on the data:",
        "",
        "| Original position | Original name | Reason |",
        "|---|---|---|",
        *[f"| {r.original_position} | `{r.original_name!r}` | {r.reason} |" for r in dropped.itertuples()],
        "",
        f"3. Added `{tb.ROW_ID}` (0-based row number in the raw CSV).",
        "4. No values were changed. No clipping, scaling, imputation, transformation, or "
        "fitting of any kind.",
        f"5. Stratified train/test split: test_size={meta['split']['test_size']}, "
        f"random_state={meta['split']['seed']}, stratified on the target only. Saved in `split.csv`.",
        "",
        "## Output",
        "",
        f"- `data.csv`: {meta['output']['rows']} rows x {meta['output']['columns']} columns "
        f"(`{tb.ROW_ID}`, `{tb.TARGET}`, {meta['output']['columns'] - 2} features)",
        "- Kept explicitly: `liability_assets_flag` (near-constant), `operating_gross_margin` and "
        "`gross_profit_to_sales` (near-identical, not identical), and all large-value features unchanged.",
        "",
        "## Target",
        "",
        "| Set | Rows | Label 0 | Label 1 |",
        "|---|---|---|---|",
        *[f"| {name} | {sum(c['count'] for c in counts.values())} | "
          f"{counts['0']['count']} ({counts['0']['share']:.2%}) | {counts['1']['count']} ({counts['1']['share']:.2%}) |"
          for name, counts in [("all", meta["target"]["counts_all"]),
                               ("train", meta["target"]["counts_train"]),
                               ("test", meta["target"]["counts_test"])]],
        "",
        f"**{LABEL_CAVEAT}**",
        "",
        f"## Features with values above 1 ({len(large)}, left unchanged; treatment pending)",
        "",
        "These are observations requiring investigation, not confirmed errors. Any treatment will be "
        "compared later using training data only.",
        "",
        "| Feature | Original name | Rows > 1 | Rows > 1e6 |",
        "|---|---|---|---|",
        *[f"| `{r['clean_name']}` | `{r['original_name']!r}` | {r['n_above_1']} | {r['n_above_1e6']} |" for r in large],
        "",
        "## Known limitations",
        "",
        "- No date column: chronological evaluation is not possible.",
        "- No company identifier: the same company could appear in both train and test.",
        "- Timing of features relative to the bankruptcy event is undocumented.",
        "- Units and definitions of the features are undocumented (UCI provides names only).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--output-dir", type=Path, help="output directory (default: interim data path from config)")
    args = parser.parse_args()

    settings = load_settings()
    raw_dir = PROJECT_ROOT / settings["paths"]["raw_data"] / "taiwan_bankruptcy"
    out_dir = args.output_dir or PROJECT_ROOT / settings["paths"]["interim_data"] / "taiwan_bankruptcy"
    seed = settings["project"]["random_seed"]
    test_size = settings["company_distress"]["test_size"]
    csv_path, meta_path = raw_dir / "data.csv", raw_dir / "metadata.json"
    out_paths = {name: out_dir / name for name in OUTPUT_FILES}

    existing = [str(p) for p in out_paths.values() if p.exists()]
    if existing and not args.force:
        print(f"Output(s) already exist: {existing}")
        print("Nothing was written. Use --force to overwrite.")
        return 1

    expected = recorded_checksum(meta_path)
    if not csv_path.exists() or expected is None:
        print(f"Missing dataset or invalid {meta_path.name} in {raw_dir}. Nothing was written.")
        return 1
    checksum_before = sha256_file(csv_path)
    if checksum_before != expected:
        print(f"Checksum mismatch for {csv_path}; expected {expected}. Nothing was written.")
        return 1

    try:
        raw = tb.read_raw(csv_path)
        processed, column_map = tb.preprocess(raw)
        split = tb.make_split(processed, test_size=test_size, seed=seed)
        data_text = tb.to_csv_text(processed)
        map_text = tb.to_csv_text(column_map)
        split_text = tb.to_csv_text(split)
    except tb.ValidationError as error:
        print(f"Validation failed: {error}")
        print("Nothing was written.")
        return 1

    if sha256_file(csv_path) != checksum_before:
        print(f"{csv_path} changed during processing. Nothing was written.")
        return 1

    split_target = processed.set_index(tb.ROW_ID).loc[split[tb.ROW_ID], tb.TARGET].to_numpy()
    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": {
            "path": csv_path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": checksum_before,
            "verified_before_and_after": True,
            "rows": int(raw.shape[0]),
            "columns": int(raw.shape[1]),
        },
        "output": {
            "rows": int(processed.shape[0]),
            "columns": int(processed.shape[1]),
            "sha256": {
                "data.csv": sha256_bytes(data_text.encode("utf-8")),
                "column_map.csv": sha256_bytes(map_text.encode("utf-8")),
                "split.csv": sha256_bytes(split_text.encode("utf-8")),
            },
        },
        "dropped_columns": [
            {"original_position": int(r.original_position), "original_name": r.original_name, "reason": r.reason}
            for r in column_map[column_map["status"] == "dropped"].itertuples()
        ],
        "target": {
            "original_name": tb.RAW_TARGET,
            "clean_name": tb.TARGET,
            "label_meaning": LABEL_CAVEAT,
            "counts_all": class_counts(processed[tb.TARGET]),
            "counts_train": class_counts(pd.Series(split_target[split["split"].to_numpy() == "train"])),
            "counts_test": class_counts(pd.Series(split_target[split["split"].to_numpy() == "test"])),
        },
        "split": {"method": "sklearn.model_selection.train_test_split, stratified on target",
                  "test_size": test_size, "seed": seed},
        "transformations": ["clean column names", "drop 4 verified columns", "add row_id", "stratified split"],
        "fitted_transformations": "none",
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    large = large_value_features(processed, column_map)
    meta["large_value_features_unchanged"] = [r["clean_name"] for r in large]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs({
        out_paths["data.csv"]: data_text,
        out_paths["column_map.csv"]: map_text,
        out_paths["split.csv"]: split_text,
        out_paths["metadata.json"]: json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        out_paths["preprocessing_report.md"]: build_report(meta, column_map, large),
    })

    if sha256_file(csv_path) != checksum_before:
        print(f"WARNING: {csv_path} checksum changed after writing outputs.")
        return 1
    print(f"Saved {len(OUTPUT_FILES)} files to {out_dir}")
    print(f"Rows: {processed.shape[0]}, columns: {processed.shape[1]}, dropped: {len(meta['dropped_columns'])}")
    print(f"Split: train {int((split['split'] == 'train').sum())}, test {int((split['split'] == 'test').sum())}")
    print("Raw CSV checksum unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
