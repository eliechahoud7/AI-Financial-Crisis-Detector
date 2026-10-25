"""Check RAG retrieval and reference answers against the accepted baselines (docs/decisions.md, D25).

Usage:
    python scripts/check_rag_regression.py            # summary; exit code 1 on any regression
    python scripts/check_rag_regression.py --json     # every check as JSON
    python scripts/check_rag_regression.py --update   # re-record the baselines after reviewing a change

Uses the development question set only (never the validation set, D24) and
the extractive answers of scripts/ask_rag.py and scripts/run_crisis_agent.py.
Needs a current RAG index and the cached embedding model. Without --update it
writes nothing; --update rewrites config/qa/rag_dev_baseline.json and
config/qa/reference_answers.json, so run it only after reviewing what changed.
"""

import argparse
import json
import sys
from dataclasses import asdict

from fincrisis.qa import integrity, rag_regression


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="print every check as JSON")
    parser.add_argument("--update", action="store_true", help="re-record the baselines (review the changes first)")
    args = parser.parse_args()
    checks = rag_regression.run(update=args.update)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2, ensure_ascii=False))
    else:
        print(integrity.summary(checks, noun="RAG regression check"))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
