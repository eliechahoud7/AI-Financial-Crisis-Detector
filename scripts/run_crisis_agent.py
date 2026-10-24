"""Ask the crisis-detector workflow a question (docs/decisions.md, D17).

Usage:
    python scripts/run_crisis_agent.py "What is the current market stress level?"
    python scripts/run_crisis_agent.py "How well do the bankruptcy models perform?"
    python scripts/run_crisis_agent.py "Which VIX value is used for date t?" --llm ollama
    python scripts/run_crisis_agent.py "..." --json      # full workflow state, including the route taken

A LangGraph workflow routes the question: current market-stress questions are
scored with the frozen market-stress models; company-distress performance
questions report the frozen held-out results; other questions go to the
document RAG; investment-advice requests are declined. Explanations are
grounded in cited project documents. Everything runs locally and read-only.
"""

import argparse
import json
import sys

import yaml

from fincrisis.agents.graph import build_graph
from fincrisis.agents.runtime import PROJECT_ROOT, make_tools


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--llm", choices=["ollama"], help="let the local Ollama model write document answers")
    parser.add_argument("--json", action="store_true", help="print the full workflow state as JSON")
    args = parser.parse_args()
    settings = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
    tools, notes = make_tools(settings, args.llm == "ollama")
    state = build_graph(tools).invoke({"question": args.question})
    state["warnings"] = [*state.get("warnings", []), *notes]
    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False, default=str))
        return 0
    print(state["answer"])
    if state.get("citations"):
        print("\nSources:")
        for c in state["citations"]:
            print(f"  [{c['marker']}] {c['source']} - {c['section']}")
    print(f"\nRoute: {' -> '.join(state['trace'])}")
    for warning in state["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
