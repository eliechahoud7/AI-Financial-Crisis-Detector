"""Run every quality check and the test suite, with one pass/fail summary (docs/decisions.md, D28).

Usage:
    python scripts/run_qa.py             # all stages; exit code 1 if any fails
    python scripts/run_qa.py --quick     # skip the unit-test suite
    python scripts/run_qa.py --live      # also run the live tests (local Ollama model, dashboard server)
    python scripts/run_qa.py --json      # every check as JSON

Stages: environment (installed packages vs requirements-lock.txt), integrity
(frozen inputs and outputs vs recorded checksums), reproducibility (today's code
vs the frozen results), rag regression (development-set retrieval and
reference answers), and tests. Everything is read-only; nothing is trained,
downloaded or rebuilt. Run it from the project root.
"""

import argparse
import json
import sys
from dataclasses import asdict

from fincrisis.qa import runner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the unit-test suite")
    parser.add_argument("--live", action="store_true", help="also run the live Ollama and dashboard-server tests")
    parser.add_argument("--json", action="store_true", help="print every check as JSON")
    args = parser.parse_args()
    if args.quick and args.live:
        parser.error("--live applies to the test suite, which --quick skips")
    stages = runner.run(runner.default_stages(live=args.live, tests=not args.quick))
    if args.json:
        print(json.dumps([{"stage": s.name, "ok": s.ok, "seconds": s.seconds,
                           "checks": [asdict(c) for c in s.checks]} for s in stages], indent=2, ensure_ascii=False))
    else:
        print(runner.report(stages))
    return 0 if all(s.ok for s in stages) else 1


if __name__ == "__main__":
    sys.exit(main())
