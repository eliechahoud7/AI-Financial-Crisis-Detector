"""Verify every frozen input and output against independently recorded checksums (docs/decisions.md, D22).

Usage:
    python scripts/verify_integrity.py           # summary; exit code 1 if any check fails
    python scripts/verify_integrity.py --json    # every check as JSON

Read-only: it hashes files and reads JSON, nothing else. Checks the manifest
(config/frozen_artifacts.json, including the RAG validation questions), the raw
market snapshot, the Taiwan raw and processed data, and the RAG index.
"""

import argparse
import json
import sys
from dataclasses import asdict

from fincrisis.qa import integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print every check as JSON")
    args = parser.parse_args()
    checks = integrity.verify_all()
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(integrity.summary(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
