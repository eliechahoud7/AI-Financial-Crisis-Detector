"""Preprocessing for the UCI Taiwanese Bankruptcy Prediction dataset.

Only deterministic, target-independent steps live here: exact parsing, name
cleaning, removal of verified redundant or constant columns, and the
stratified train/test split. Nothing is fitted to the data
(see docs/decisions.md, D4 and D8).
"""

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_TARGET = "Bankrupt?"
TARGET = "bankrupt"
ROW_ID = "row_id"
EXPECTED_ROWS = 6819
EXPECTED_COLUMNS = 96
COMPLEMENT_TOLERANCE = 1e-12

# Verified on the raw CSV. Names exactly as stored, including the leading space.
IDENTICAL_DROPS = {  # dropped column -> kept column with identical values
    " Current Liability to Liability": " Current Liabilities/Liability",
    " Current Liability to Equity": " Current Liabilities/Equity",
}
COMPLEMENT_DROPS = {  # dropped column -> kept column; dropped == 1 - kept
    " Net worth/Assets": " Debt ratio %",
}
CONSTANT_DROPS = [" Net Income Flag"]

# Explicit replacements; any other non-ASCII character is rejected.
NAME_REPLACEMENTS = {"¥": " ", "%": " pct "}


class ValidationError(Exception):
    """An assumption about the data did not hold; nothing should be written."""


def read_raw(path: Path) -> pd.DataFrame:
    with Path(path).open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    df = pd.read_csv(path, encoding="utf-8", float_precision="round_trip")
    if list(df.columns) != header:
        raise ValidationError("pandas changed the column names (duplicate names in header?)")
    if df.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValidationError(f"Expected {EXPECTED_ROWS} x {EXPECTED_COLUMNS}, found {df.shape}")
    return df


def clean_name(name: str) -> str:
    if name == RAW_TARGET:
        return TARGET
    text = name
    for old, new in NAME_REPLACEMENTS.items():
        text = text.replace(old, new)
    if not text.isascii():
        raise ValidationError(f"Unhandled non-ASCII character in column name {name!r}")
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_").lower()
    if not text:
        raise ValidationError(f"Column name {name!r} is empty after cleaning")
    return text


def clean_names(names: list[str]) -> dict[str, str]:
    mapping = {name: clean_name(name) for name in names}
    seen: dict[str, str] = {}
    for original, cleaned in mapping.items():
        if cleaned == ROW_ID or cleaned in seen:
            other = seen.get(cleaned, ROW_ID)
            raise ValidationError(f"Name collision: {original!r} and {other!r} both become {cleaned!r}")
        seen[cleaned] = original
    return mapping


def validate_target(df: pd.DataFrame) -> None:
    if RAW_TARGET not in df.columns:
        raise ValidationError(f"Target column {RAW_TARGET!r} not found")
    target = df[RAW_TARGET]
    if target.isna().any() or set(target.unique().tolist()) != {0, 1}:
        raise ValidationError(f"Target must contain exactly 0 and 1, found {sorted(target.unique().tolist())}")


def validate_drops(df: pd.DataFrame) -> dict[str, str]:
    """Re-verify every drop against the data; return {dropped column: reason}."""
    reasons = {}
    for drop, keep in IDENTICAL_DROPS.items():
        _require_columns(df, drop, keep)
        if df[drop].dtype != df[keep].dtype or not df[drop].equals(df[keep]):
            raise ValidationError(f"{drop!r} is no longer identical to {keep!r}")
        reasons[drop] = f"identical to {keep.strip()!r}"
    for drop, keep in COMPLEMENT_DROPS.items():
        _require_columns(df, drop, keep)
        deviation = float((df[drop] + df[keep] - 1).abs().max())
        if not deviation <= COMPLEMENT_TOLERANCE:
            raise ValidationError(f"{drop!r} + {keep!r} deviates from 1 by {deviation:g}")
        reasons[drop] = f"equals 1 - {keep.strip()!r} (max deviation {deviation:.1e})"
    for drop in CONSTANT_DROPS:
        _require_columns(df, drop)
        if df[drop].nunique(dropna=False) != 1:
            raise ValidationError(f"{drop!r} is no longer constant")
        reasons[drop] = f"constant (always {df[drop].iloc[0]})"
    return reasons


def _require_columns(df: pd.DataFrame, *names: str) -> None:
    missing = [name for name in names if name not in df.columns]
    if missing:
        raise ValidationError(f"Expected column(s) not found: {missing}")


def preprocess(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (processed data, column map). The input frame is not modified."""
    validate_target(raw)
    reasons = validate_drops(raw)
    mapping = clean_names(list(raw.columns))

    kept = [c for c in raw.columns if c not in reasons]
    processed = raw[kept].rename(columns=mapping)
    processed.insert(0, ROW_ID, np.arange(len(raw)))

    column_map = pd.DataFrame({
        "original_position": range(len(raw.columns)),
        "original_name": raw.columns,
        "clean_name": [mapping[c] for c in raw.columns],
        "status": ["dropped" if c in reasons else "kept" for c in raw.columns],
        "reason": [reasons.get(c, "") for c in raw.columns],
    })
    return processed, column_map


def make_split(processed: pd.DataFrame, test_size: float, seed: int) -> pd.DataFrame:
    """Stratified train/test assignment by row_id; uses only the target for stratification."""
    train_ids, test_ids = train_test_split(
        processed[ROW_ID], test_size=test_size, stratify=processed[TARGET], random_state=seed
    )
    split = pd.concat([
        pd.DataFrame({ROW_ID: train_ids, "split": "train"}),
        pd.DataFrame({ROW_ID: test_ids, "split": "test"}),
    ])
    return split.sort_values(ROW_ID).reset_index(drop=True)


def processed_checksums(directory: Path) -> dict[str, str]:
    """SHA-256 of the processed files listed in metadata.json, verified against it."""
    directory = Path(directory)
    recorded = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))["output"]["sha256"]
    actual = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in recorded}
    mismatched = [name for name in recorded if actual[name] != recorded[name]]
    if mismatched:
        raise ValidationError(f"Processed file(s) do not match metadata.json: {mismatched}")
    return actual


def load_processed(directory: Path, subset: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (features, target, row_id) for the saved 'train' or 'test' split.

    Checksums are verified first. Only rows of the requested subset are returned,
    so code that asks for 'train' never receives test rows.
    """
    if subset not in ("train", "test"):
        raise ValueError(f"subset must be 'train' or 'test', not {subset!r}")
    directory = Path(directory)
    processed_checksums(directory)
    data = pd.read_csv(directory / "data.csv", float_precision="round_trip")
    split = pd.read_csv(directory / "split.csv")
    if not data[ROW_ID].equals(split[ROW_ID]):
        raise ValidationError("data.csv and split.csv row_ids do not match")
    if not set(split["split"]) <= {"train", "test"}:
        raise ValidationError(f"Unexpected split labels: {sorted(set(split['split']))}")
    rows = data[split["split"].to_numpy() == subset]
    features = rows.drop(columns=[ROW_ID, TARGET]).set_index(rows[ROW_ID])
    return features, rows[TARGET].set_axis(rows[ROW_ID]), rows[ROW_ID].reset_index(drop=True)


def to_csv_text(df: pd.DataFrame) -> str:
    """Serialize to CSV and confirm the text parses back to identical values."""
    text = df.to_csv(index=False, lineterminator="\n")
    parsed = pd.read_csv(io.StringIO(text), float_precision="round_trip", keep_default_na=False)
    numeric = df.select_dtypes("number").columns
    if list(parsed.columns) != list(df.columns) or not parsed[numeric].equals(df[numeric]):
        raise ValidationError("CSV serialization does not round-trip exactly")
    return text
