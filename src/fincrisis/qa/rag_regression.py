"""RAG regression checks: development-set retrieval and reference answers (docs/decisions.md, D25).

Documents are edited often, and every edit changes the index. Two baselines in
`config/qa/` record the accepted state, and these checks compare the current
index and code with them:

- retrieval on the development question set (never the validation set: the
  question file is resolved through the validation guard, D24). For the
  configured retriever, no summary metric may fall below its baseline; the
  other retrievers are reported for information only;
- reference answers: the RAG `answer()` call of `scripts/ask_rag.py` and the
  LangGraph workflow of `scripts/run_crisis_agent.py`, for fixed questions.
  Answers, citations and routes must match exactly. Warnings that depend on
  today's date (the stale-snapshot warning) are left out.

A legitimate change (a new document, a decision record) is accepted by
re-recording the baselines with `scripts/check_rag_regression.py --update`,
after reviewing the differences it prints. Nothing is tuned here: the
baselines record results, they do not choose settings.
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fincrisis.dashboard import loaders
from fincrisis.qa import validation_guard as guard
from fincrisis.qa.integrity import Check
from fincrisis.rag import answer as ans
from fincrisis.rag import evaluation as ev
from fincrisis.rag import pipeline

QA_DIR = loaders.PROJECT_ROOT / "config" / "qa"
RETRIEVAL_BASELINE = QA_DIR / "rag_dev_baseline.json"
ANSWER_BASELINE = QA_DIR / "reference_answers.json"
# Fixed questions, each taken from the development set or the README (a test checks this), so none can be a
# validation question.
ANSWER_QUESTIONS = ["Which VIX value is used for date t?",
                    "How many trading days ahead does the market-stress label look?",
                    "Who won the 2018 FIFA World Cup?"]
WORKFLOW_QUESTIONS = ["What is the current market stress level?", "How well do the bankruptcy models perform?",
                      "Which VIX value is used for date t?", "Should I sell my stocks now?",
                      "Who won the 2018 FIFA World Cup?"]
DATE_DEPENDENT = re.compile(r"snapshot is \d+ days old")


def development_questions(rag: dict, root: Path = loaders.PROJECT_ROOT) -> list[dict]:
    path, _ = guard.resolve_question_set(None, rag["eval_set"], root)  # the development set only
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def retrieval(store, rag: dict, questions: list[dict]) -> dict:
    """Per-mode summary metrics and per-question first-relevant ranks on the development set."""
    out = {}
    for mode in pipeline.retrieval_modes(rag):
        retriever = pipeline.make_retriever(store, rag, mode)
        result = ev.evaluate(retriever.search, questions, rag["eval_k"], rag["min_relevance"], retriever.max_dense_score)
        out[mode] = {"summary": {k: round(v, 6) for k, v in result["summary"].items() if isinstance(v, float)},
                     "ranks": {r["id"]: r["rank"] for r in result["results"]}}
    return out


def compare_retrieval(current: dict, baseline: dict, configured: str) -> list[Check]:
    checks = []
    for mode, now in current.items():
        then = baseline["modes"].get(mode)
        if then is None:  # a mode added since the baseline was recorded: review it, then re-record with --update
            status = "mismatch" if mode == configured else "ok"
            checks.append(Check(f"rag retrieval ({mode}{', configured' if mode == configured else ''})",
                                "baseline", status, "no baseline recorded for this mode"))
            continue
        moved = sorted((q, then["ranks"].get(q), r) for q, r in now["ranks"].items() if then["ranks"].get(q) != r)
        detail = "rank changes (question, baseline, now): " + (", ".join(map(str, moved)) or "none")
        for metric, value in now["summary"].items():
            before = then["summary"].get(metric)
            worse = before is not None and value < before - 1e-9
            status = "mismatch" if worse and mode == configured else "ok"
            note = f"{value:.3f} vs baseline {before:.3f}" + ("; " + detail if worse else "")
            checks.append(Check(f"rag retrieval ({mode}{', configured' if mode == configured else ''})",
                                metric, status, note))
    return checks


def _answer_record(result: dict) -> dict:
    return {"status": result["status"], "method": result["method"], "answer": result["answer"],
            "citations": [{k: c[k] for k in ("marker", "source", "section", "chunk_id")} for c in result["citations"]],
            "top_score": None if result["top_score"] is None else round(result["top_score"], 6)}


def _workflow_record(state: dict) -> dict:
    return {"route": state["route"], "trace": state["trace"], "answer": state["answer"],
            "citations": [{k: c[k] for k in ("marker", "source", "section") if k in c} for c in state.get("citations", [])],
            "warnings": [w for w in state.get("warnings", []) if not DATE_DEPENDENT.search(w)]}


def reference_answers(store, embeddings, settings: dict) -> dict:
    """Current answers to the fixed questions, through the same calls as the two CLIs (extractive)."""
    from fincrisis.agents.graph import build_graph
    from fincrisis.agents.runtime import make_tools
    rag = settings["rag"]
    retriever = pipeline.make_retriever(store, rag)
    answers = {q: _answer_record(ans.answer(q, store, embeddings, rag["top_k"], rag["min_relevance"],
                                            rag["max_answer_sentences"], None, rag["ollama"]["min_support"], retriever))
               for q in ANSWER_QUESTIONS}
    tools, _ = make_tools(settings, use_llm=False)
    graph = build_graph(tools)
    workflow = {q: _workflow_record(graph.invoke({"question": q})) for q in WORKFLOW_QUESTIONS}
    return {"answer": answers, "workflow": workflow}


def compare_answers(current: dict, baseline: dict) -> list[Check]:
    checks = []
    for kind in ("answer", "workflow"):
        for question, now in current[kind].items():
            then = baseline["answers"][kind].get(question)
            if then is None:
                checks.append(Check(f"reference answers ({kind})", question, "missing", "no baseline"))
                continue
            changed = [field for field in now if now[field] != then.get(field)]
            checks.append(Check(f"reference answers ({kind})", question, "mismatch" if changed else "ok",
                                f"changed: {', '.join(changed)}" if changed else ""))
    return checks


def _state(settings: dict) -> dict:
    status = loaders.rag_index_status(settings["rag"])
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=loaders.PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip() or None
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
            "index_built_at_utc": status["built_at_utc"], "index_chunks": status["chunks"],
            "embedding_model": settings["rag"]["embedding_model"], "eval_set": settings["rag"]["eval_set"]}


def run(settings: dict | None = None, update: bool = False) -> list[Check]:
    """Compare with the baselines, or (update=True) re-record them; the index must be current."""
    settings = settings or loaders.load_settings()
    rag = settings["rag"]
    status = loaders.rag_index_status(rag)
    if not status["ready"]:
        return [Check("rag regression", "index", "missing", f"{status['problem']}. {status['remedy']}")]
    store, _, embeddings = pipeline.open_store(rag)
    current_retrieval = retrieval(store, rag, development_questions(rag))
    current_answers = reference_answers(store, embeddings, settings)
    if update:
        QA_DIR.mkdir(parents=True, exist_ok=True)
        previous = json.loads(RETRIEVAL_BASELINE.read_text(encoding="utf-8")) if RETRIEVAL_BASELINE.exists() else {}
        RETRIEVAL_BASELINE.write_text(json.dumps({
            **{k: v for k, v in previous.items() if k == "reference_d16"}, **_state(settings),
            "configured_mode": rag["retrieval"]["mode"], "modes": current_retrieval}, indent=2) + "\n",
            encoding="utf-8", newline="\n")
        ANSWER_BASELINE.write_text(json.dumps({**_state(settings), "answers": current_answers}, indent=2,
                                              ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        return [Check("rag regression", "baselines", "ok", f"re-recorded in {QA_DIR}")]
    try:
        retrieval_baseline = json.loads(RETRIEVAL_BASELINE.read_text(encoding="utf-8"))
        answer_baseline = json.loads(ANSWER_BASELINE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [Check("rag regression", "baselines", "missing", str(error))]
    return (compare_retrieval(current_retrieval, retrieval_baseline, rag["retrieval"]["mode"])
            + compare_answers(current_answers, answer_baseline))
