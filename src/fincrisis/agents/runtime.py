"""Connect the workflow to the real project components (docs/decisions.md, D17).

`make_tools` is shared by `scripts/run_crisis_agent.py` and the dashboard, so
both run the same read-only tools with the same settings. The RAG index (and
the local model, if requested) is loaded lazily, only when a branch needs
documents.
"""

from pathlib import Path

from fincrisis.agents import tools as agent_tools
from fincrisis.agents.graph import AgentTools
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline

PROJECT_ROOT = pipeline.PROJECT_ROOT


def make_tools(settings: dict, use_llm: bool, root: Path = PROJECT_ROOT) -> tuple[AgentTools, list[str]]:
    """The workflow's tools, plus a list that collects notes (e.g. "local model unavailable") as they occur."""
    rag, agent, paths = settings["rag"], settings["agent"], settings["paths"]
    notes = []
    state = {}

    def load():
        if "store" not in state:  # load the index only if a branch needs documents
            state["store"], _, state["embeddings"] = pipeline.open_store(rag, root)
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
        assess_market=lambda: agent_tools.assess_market(root, paths, agent["stale_after_days"]),
        distress_summary=lambda: agent_tools.distress_summary(root, paths),
        rag_answer=rag_answer,
        context_answer=context_answer,
    ), notes
