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

from fincrisis.agents import tools as agent_tools
from fincrisis.agents.graph import AgentTools, build_graph
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline

PROJECT_ROOT = pipeline.PROJECT_ROOT


def make_tools(settings: dict, use_llm: bool) -> tuple[AgentTools, list[str]]:
    rag, agent, paths = settings["rag"], settings["agent"], settings["paths"]
    notes = []
    state = {}

    def load():
        if "store" not in state:  # load the index only if a branch needs documents
            state["store"], _, state["embeddings"] = pipeline.open_store(rag)
            state["retriever"] = pipeline.make_retriever(state["store"], rag)
            state["generator"] = None
            if use_llm:
                o = rag["ollama"]
                status = ans.ollama_status(o["url"], o["model"])
                if status["available"]:
                    state["generator"] = ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"],
                                                              rag["max_answer_sentences"])
                else:
                    notes.append(f"local model unavailable ({status['reason']}); extractive answers used")

    def rag_answer(question: str) -> dict:
        load()
        return ans.answer(question, state["store"], state["embeddings"], rag["top_k"], rag["min_relevance"],
                          rag["max_answer_sentences"], state["generator"], rag["ollama"]["min_support"],
                          state["retriever"])

    def context_answer(branch: str) -> dict:
        """Extractive, one sentence per configured entry, so the explanation is deterministic and quotes the docs."""
        load()
        return agent_tools.context_answer(agent["context"][branch], lambda question, scope: ans.answer(
            question, state["store"], state["embeddings"], rag["top_k"], rag["min_relevance"], 1, scope=scope))

    return AgentTools(
        assess_market=lambda: agent_tools.assess_market(PROJECT_ROOT, paths, agent["stale_after_days"]),
        distress_summary=lambda: agent_tools.distress_summary(PROJECT_ROOT, paths),
        rag_answer=rag_answer,
        context_answer=context_answer,
    ), notes


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
