"""LangGraph workflow connecting market analysis, company-distress results, and document retrieval (D17).

    START -> route_request -+- advice    -> refuse_advice ------------------------------> END
                            +- market    -> assess_market -+- error -> report_error -----> END
                            |                              +- ok    -> retrieve_context -+
                            +- distress  -> summarize_distress -----> retrieve_context -+
                            +- documents ---------------------------> retrieve_context -+
                                       retrieve_context -+- error (documents)       -> report_error -> END
                                                         +- no evidence (documents) -> abstain -> END
                                                         +- otherwise               -> compose_answer -> END

The graph only orchestrates. Its tools are injected (`AgentTools`), so the
nodes reuse the existing market-stress models, the frozen distress results,
and the RAG `answer()` function without copying them, and tests can replace
them with fakes. Routing is deterministic (keyword rules), not model-driven.
"""

import re
from dataclasses import dataclass
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

DISCLAIMER = ("Research output only; not a forecast or financial advice. Results describe 1990-2026 history and "
              "carry the limitations documented in docs/decisions.md and docs/model_card.md.")
REFUSAL = ("I can't give investment or trading advice. I can report the project's market-stress signal, the "
           "company-distress model results, or what the project documents say.")

ADVICE = re.compile(r"\b(should|shall|would you|do you recommend|is it (a )?good (time|idea)|recommend)\b"
                    r".*\b(buy\w*|sell\w*|invest\w*|short\w*|hold\w*|trad\w*)\b|\binvestment advice\b|\bwhich stocks?\b", re.I)
MARKET = re.compile(r"\b(market|s&p|sp500|spx|vix|volatility|drawdown|stress|sell-?off|crash)\b", re.I)
CURRENT = re.compile(r"\b(current(ly)?|now|today|latest|at the moment|this week|right now|signal|level|elevated)\b", re.I)
DISTRESS = re.compile(r"\b(bankrupt\w*|distress|insolven\w*|compan(y|ies)|taiwan\w*)\b", re.I)
RESULTS = re.compile(r"\b(how (well|good|accurate)|results?|performance|perform\w*|accura\w*|score\w*|metrics?)\b", re.I)


class AgentState(TypedDict, total=False):
    question: str
    route: str
    market: dict
    distress: dict
    rag: dict
    answer: str
    citations: list
    warnings: list
    error: str
    trace: list


@dataclass
class AgentTools:
    """Callables the workflow uses; each wraps an existing project component.

    `rag_answer(question)` answers a document question from the whole index.
    `context_answer(branch)` returns the cited explanation configured for the
    "market" or "distress" branch (`agent.context` in config).
    """
    assess_market: Callable[[], dict]
    distress_summary: Callable[[], dict]
    rag_answer: Callable[[str], dict]
    context_answer: Callable[[str], dict]


def route(question: str) -> str:
    if ADVICE.search(question):
        return "advice"
    if MARKET.search(question) and CURRENT.search(question):
        return "market"
    if DISTRESS.search(question) and RESULTS.search(question):
        return "distress"
    return "documents"


def build_graph(tools: AgentTools):
    def step(state: AgentState, name: str) -> list:
        return [*state.get("trace", []), name]

    def route_request(state: AgentState) -> AgentState:
        return {"route": route(state["question"]), "warnings": [], "trace": step(state, "route_request")}

    def refuse_advice(state: AgentState) -> AgentState:
        return {"answer": REFUSAL, "citations": [], "trace": step(state, "refuse_advice")}

    def assess_market(state: AgentState) -> AgentState:
        try:
            market = tools.assess_market()
        except Exception as error:  # a missing or altered frozen output must stop this branch, not the program
            return {"error": f"market assessment unavailable: {error}", "trace": step(state, "assess_market")}
        return {"market": market, "warnings": [*state["warnings"], *market.get("warnings", [])],
                "trace": step(state, "assess_market")}

    def summarize_distress(state: AgentState) -> AgentState:
        try:
            return {"distress": tools.distress_summary(), "trace": step(state, "summarize_distress")}
        except Exception as error:  # report it in the answer; the documents can still be cited
            return {"distress": {"unavailable": str(error)},
                    "warnings": [*state["warnings"], f"distress results unavailable: {error}"],
                    "trace": step(state, "summarize_distress")}

    def retrieve_context(state: AgentState) -> AgentState:
        scope = state["route"] if state["route"] in ("market", "distress") else None
        try:
            rag = tools.context_answer(scope) if scope else tools.rag_answer(state["question"])
        except Exception as error:  # e.g. a stale or missing index: never crash the workflow
            update = {"rag": {"status": "unavailable", "answer": "", "citations": [], "warnings": []},
                      "warnings": [*state["warnings"], f"document retrieval unavailable: {error}"],
                      "trace": step(state, "retrieve_context")}
            if scope is None:
                update["error"] = f"document retrieval unavailable: {error}"
            return update
        return {"rag": rag, "warnings": [*state["warnings"], *rag.get("warnings", [])],
                "trace": step(state, "retrieve_context")}

    def abstain(state: AgentState) -> AgentState:
        return {"answer": state["rag"]["answer"], "citations": [], "trace": step(state, "abstain")}

    def report_error(state: AgentState) -> AgentState:
        return {"answer": f"The request could not be completed: {state['error']}. Nothing was estimated in its place.",
                "citations": [], "trace": step(state, "report_error")}

    def compose_answer(state: AgentState) -> AgentState:
        parts = []
        if state.get("market"):
            parts.append(market_text(state["market"]))
        if state.get("distress"):
            d = state["distress"]
            parts.append(f"Company-distress results are unavailable: {d['unavailable']}" if "unavailable" in d
                         else distress_text(d))
        rag = state["rag"]
        if rag["status"] == "answered":
            heading = "From the project documents:" if parts else None
            parts.append(f"{heading}\n{rag['answer']}" if heading else rag["answer"])
        elif rag["status"] == "unavailable":
            parts.append("Document context is unavailable (see warnings).")
        elif parts:
            parts.append("No supporting passage was found in the project documents.")
        parts.append(DISCLAIMER)
        return {"answer": "\n\n".join(parts), "citations": rag.get("citations", []), "trace": step(state, "compose_answer")}

    graph = StateGraph(AgentState)
    for name, node in [("route_request", route_request), ("refuse_advice", refuse_advice),
                       ("assess_market", assess_market), ("summarize_distress", summarize_distress),
                       ("retrieve_context", retrieve_context), ("abstain", abstain),
                       ("report_error", report_error), ("compose_answer", compose_answer)]:
        graph.add_node(name, node)
    graph.add_edge(START, "route_request")
    graph.add_conditional_edges("route_request", lambda s: s["route"],
                                {"advice": "refuse_advice", "market": "assess_market",
                                 "distress": "summarize_distress", "documents": "retrieve_context"})
    graph.add_conditional_edges("assess_market", lambda s: "error" if s.get("error") else "ok",
                                {"error": "report_error", "ok": "retrieve_context"})
    graph.add_edge("summarize_distress", "retrieve_context")
    graph.add_conditional_edges("retrieve_context", after_retrieval,
                                {"error": "report_error", "abstain": "abstain", "compose": "compose_answer"})
    for terminal in ("refuse_advice", "report_error", "abstain", "compose_answer"):
        graph.add_edge(terminal, END)
    return graph.compile()


def after_retrieval(state: AgentState) -> str:
    """Documents branch: an error or missing evidence ends the run; other branches always compose."""
    if state["route"] != "documents":
        return "compose"
    if state.get("error"):
        return "error"
    return "compose" if state["rag"]["status"] == "answered" else "abstain"


def answer_method(state: AgentState, use_llm: bool) -> dict | None:
    """How the run's document text was produced, from the RAG result in the state (not from the answer text).

    Returns {"kind", "text"} with kind "generated", "rejected" (the model's answer failed the grounding check
    and the extractive answer is shown), "extractive" or "abstained"; None for refused or failed runs. Market
    and distress context is always extractive: the local model is used for document questions only.
    """
    rag = state.get("rag")
    if state.get("route") == "advice" or rag is None or rag.get("status") == "unavailable":
        return None
    if state["route"] != "documents":
        if rag.get("status") != "answered":
            return {"kind": "abstained", "text": "no supporting passage reached the relevance threshold"}
        return {"kind": "extractive", "text": "extractive document context (the local model is used for document "
                                              "questions only)"}
    generation = rag.get("generation")
    if rag["status"] != "answered":
        if rag.get("method") == "generative":
            return {"kind": "abstained", "text": f"abstained: {generation['model']} found the evidence insufficient"}
        best = "" if rag.get("top_score") is None else f" (best {rag['top_score']:.3f}, needs {rag['min_relevance']:.2f})"
        return {"kind": "abstained", "text": f"abstained: no passage reached the relevance threshold{best}"}
    if rag["method"] == "generative":
        repairs = len(generation.get("citation_repairs") or [])
        return {"kind": "generated", "text": f"generated by {generation['model']} ({generation.get('seconds')} s), "
                                             "passed the grounding check"
                                             + (f"; {repairs} citation marker(s) repaired" if repairs else "")}
    if generation is not None and generation.get("grounding_problems"):
        return {"kind": "rejected", "text": f"extractive fallback: {generation['model']}'s answer was rejected by the "
                                            f"grounding check ({generation['grounding_problems'][0]})"}
    return {"kind": "extractive", "text": "extractive: the local model could not be used (see warnings)" if use_llm
            else "extractive (local model off)"}


def market_text(m: dict) -> str:
    p = m["primary"]
    lines = [f"Market-stress signal as of {m['as_of']} (label: {m['label_definition']}).",
             f"- Primary model `{p['name']}` (chosen on validation only): score {p['score']:.4f} vs frozen threshold "
             f"{p['threshold']:.4f} -> {'ELEVATED' if p['elevated'] else 'not elevated'}."]
    if "historical_percentile" in p:
        lines.append(f"- The score is at the {p['historical_percentile']:.0%} percentile of {p['history_rows']} "
                     "historical trading days.")
    if m.get("secondary"):  # D35: a fitted model's frozen threshold is not transferable, so no verdict is given
        s = m["secondary"]
        lines.append(f"- Secondary `{s['name']}`: score {s['score']:.3f} (uncalibrated; not a probability; for "
                     "comparison only, with no decision threshold).")
    lines.append(f"- Measured reliability on the held-out test ({m['test_period']['first_date']} to "
                 f"{m['test_period']['last_date']}): average precision {p['test_average_precision']:.3f} vs a "
                 f"base rate of {m['test_prevalence']:.3f}; at the threshold, precision {p['test_precision_at_threshold']:.2f} "
                 f"and recall {p['test_recall_at_threshold']:.2f}. The signal is weak and produces many false alarms.")
    return "\n".join(lines)


def distress_text(d: dict) -> str:
    lines = [f"Company-distress baselines, frozen held-out test ({d['test_rows']} rows, {d['test_label_1']} with label 1):"]
    for family, r in d["families"].items():
        lo, hi = r["average_precision_95ci"]
        lines.append(f"- {family} (`{r['config']}`): average precision {r['average_precision']:.3f} "
                     f"[{lo:.3f}, {hi:.3f}], ROC-AUC {r['roc_auc']:.3f}.")
    lines.append(f"- {d['label_caveat']}")
    return "\n".join(lines)
