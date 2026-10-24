"""Check that today's code still reproduces the frozen results (docs/decisions.md, D23).

Usage:
    python scripts/check_reproducibility.py           # summary; exit code 1 on any difference
    python scripts/check_reproducibility.py --json    # every check as JSON

Read-only: rebuilds the market features and target from the frozen snapshot,
re-scores the frozen test rows with the saved models, and recomputes the
latest-day signal, then compares each with the frozen values. Nothing is fitted
or written.
"""

import argparse
import json
import sys
from dataclasses import asdict

from fincrisis.qa import drift, integrity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print every check as JSON")
    args = parser.parse_args()
    checks = drift.verify_all()
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        print(integrity.summary(checks, noun="reproducibility check"))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
