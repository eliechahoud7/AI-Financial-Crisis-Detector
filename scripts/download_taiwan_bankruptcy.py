"""Download the UCI Taiwanese Bankruptcy Prediction dataset into data/raw/.

Usage:
    python scripts/download_taiwan_bankruptcy.py           # skip if already present
    python scripts/download_taiwan_bankruptcy.py --force   # re-download and overwrite

Source:  https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction
License: CC BY 4.0, doi:10.24432/C5004D

The target column "Bankrupt?" is checked to contain only 0 and 1. UCI does not
document what each value means; see docs/decisions.md (D4).
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
ARCHIVE_MEMBER = "data.csv"
TARGET = "Bankrupt?"
EXPECTED_ROWS = 6819
EXPECTED_COLUMNS = 96  # target + 95 features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    return PROJECT_ROOT / settings["paths"]["raw_data"] / "taiwan_bankruptcy"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fincrisis-research/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def extract_csv(zip_bytes: bytes) -> bytes:
    # Read one named member only; never extractall() on downloaded archives.
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        if ARCHIVE_MEMBER not in archive.namelist():
            raise ValueError(f"{ARCHIVE_MEMBER} not in archive: {archive.namelist()}")
        return archive.read(ARCHIVE_MEMBER)


def validate(csv_bytes: bytes) -> dict:
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    if not rows:
        raise ValueError("CSV is empty")
    header, body = [name.strip() for name in rows[0]], rows[1:]
    if len(header) != EXPECTED_COLUMNS or header[0] != TARGET:
        raise ValueError(f"Unexpected header: {len(header)} columns, first={header[0]!r}")
    if len(body) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(body)}")
    if any(len(row) != EXPECTED_COLUMNS for row in body):
        raise ValueError("Some rows have the wrong number of columns")
    labels = Counter(row[0].strip() for row in body)
    if set(labels) != {"0", "1"}:
        raise ValueError(f"Target values must be exactly 0 and 1, found: {dict(labels)}")
    return {"rows": len(body), "columns": len(header), "target_value_counts": dict(sorted(labels.items()))}


def recorded_checksum(meta_path: Path) -> str | None:
    """Return the CSV checksum stored in metadata.json, or None if missing or invalid."""
    try:
        value = json.loads(meta_path.read_text(encoding="utf-8"))["csv_sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) else None


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing download")
    args = parser.parse_args()

    out_dir = output_dir()
    csv_path, meta_path = out_dir / "data.csv", out_dir / "metadata.json"

    # Existing data is never touched without --force, whatever state the metadata is in.
    if csv_path.exists() and not args.force:
        expected = recorded_checksum(meta_path)
        if expected is None:
            print(f"Already present: {csv_path}, but {meta_path.name} is missing or invalid.")
            print("The file was left unchanged. Use --force to re-download and overwrite it.")
            return 1
        if expected != sha256(csv_path.read_bytes()):
            print(f"Already present: {csv_path}, but its checksum does not match {meta_path.name}.")
            print("The file was left unchanged. Use --force to re-download and overwrite it.")
            return 1
        print(f"Already present: {csv_path} (checksum OK). Use --force to re-download.")
        return 0

    print(f"Downloading {URL}")
    zip_bytes = download(URL)
    csv_bytes = extract_csv(zip_bytes)
    summary = validate(csv_bytes)  # validate before writing anything

    out_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(csv_path, csv_bytes)
    metadata = {
        "source_url": URL,
        "doi": "10.24432/C5004D",
        "license": "CC BY 4.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "zip_sha256": sha256(zip_bytes),
        "csv_sha256": sha256(csv_bytes),
        "target_column": TARGET,
        "target_label_meaning": "not documented by UCI; see docs/decisions.md (D4)",
        **summary,
    }
    write_atomic(meta_path, json.dumps(metadata, indent=2).encode("utf-8"))
    print(f"Saved {csv_path}")
    print(f"Rows: {summary['rows']}, columns: {summary['columns']}, "
          f"target value counts: {summary['target_value_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
