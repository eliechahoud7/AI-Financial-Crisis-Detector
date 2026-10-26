"""Connect the workflow to the real project components (docs/decisions.md, D17).

`make_tools` is shared by `scripts/run_crisis_agent.py` and the dashboard, so
both run the same read-only tools with the same settings. The RAG index is
loaded lazily, only when a branch needs documents. The local model, if
requested, is used for document questions only; while it is unavailable, each
document question checks again, so a workflow created before Ollama started
recovers without being rebuilt.
"""

from pathlib import Path

from fincrisis.agents import tools as agent_tools
from fincrisis.agents.graph import AgentTools
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline

PROJECT_ROOT = pipeline.PROJECT_ROOT


def make_tools(settings: dict, use_llm: bool, root: Path = PROJECT_ROOT) -> tuple[AgentTools, list[str]]:
    """The workflow's tools, plus a list of notes (e.g. "local model unavailable") that reflects the latest check."""
    rag, agent, paths = settings["rag"], settings["agent"], settings["paths"]
    notes = []
    state = {"generator": None}

    def load():
        if "store" not in state:  # load the index only if a branch needs documents
            state["store"], _, state["embeddings"] = pipeline.open_store(rag, root)
            state["retriever"] = pipeline.make_retriever(state["store"], rag)

    def generator():
        """The Ollama generator, checking the server again while it is unavailable; None without --llm."""
        if use_llm and state["generator"] is None:
            o = rag["ollama"]
            status = ans.ollama_status(o["url"], o["model"])
            notes.clear()  # the note describes the latest check only
            if status["available"]:
                state["generator"] = ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"],
                                                          rag["max_answer_sentences"])
            else:
                notes.append(f"local model unavailable ({status['reason']}); extractive answers used")
        return state["generator"]

    def rag_answer(question: str) -> dict:
        load()
        return ans.answer(question, state["store"], state["embeddings"], rag["top_k"], rag["min_relevance"],
                          rag["max_answer_sentences"], generator(), rag["ollama"]["min_support"],
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
