"""Evaluate generated (Ollama) answers on the labelled question set (docs/decisions.md, D15).

Usage:
    python scripts/evaluate_rag_generation.py            # stop if a report exists
    python scripts/evaluate_rag_generation.py --force

For every question, records whether the local model's answer was accepted
(passed the grounding checks), rejected (extractive answer used instead),
or replaced by "insufficient evidence" (by the retrieval threshold or by the
model itself). For accepted answers, it checks whether at least one cited
passage is a relevant passage for the question. Accepted answers are listed
in the report for manual review: the automatic checks are lexical and do not
prove an answer is correct. Writes data/reports/rag/generation_eval.json and .md.
"""

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone

from fincrisis.rag import answer as ans
from fincrisis.rag import evaluation as ev
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline


def outcome(result: dict) -> str:
    if result["method"] == "none":
        return "abstained_threshold"
    if result["status"] == "insufficient_evidence":
        return "abstained_model"
    if result["method"] == "generative":
        return "accepted"
    return "rejected" if result.get("generation") else "generator_error"


def evaluate(questions, store, embeddings, settings, generator) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        r = ans.answer(q["question"], store, embeddings, settings["top_k"], settings["min_relevance"],
                       settings["max_answer_sentences"], generator, settings["ollama"]["min_support"])
        g = r.get("generation") or {}
        answerable = q.get("answerable", True)
        cited = [e for e in r["evidence"] if e["rank"] in {c["marker"] for c in r["citations"]}]
        row = {"id": q["id"], "question": q["question"], "answerable": answerable, "outcome": outcome(r),
               "top_score": r["top_score"], "seconds": g.get("seconds"), "raw_answer": g.get("raw_answer"),
               "grounding_problems": g.get("grounding_problems"), "final_answer": r["answer"],
               "cites_relevant": (any(ev.is_relevant(e, q["relevant"]) for e in cited)
                                  if answerable and r["method"] == "generative" else None),
               "citations": [f"{c['source']} - {c['section']}" for c in r["citations"]], "warnings": r["warnings"]}
        rows.append(row)
        print(f"  {i:2}/{len(questions)} {q['id']} {row['outcome']:20} {row['seconds'] or '-'} s", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    def share(subset, key, value):
        return round(sum(r[key] == value for r in subset) / len(subset), 4) if subset else None
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    accepted = [r for r in answerable if r["outcome"] == "accepted"]
    seconds = [r["seconds"] for r in rows if r["seconds"]]
    return {
        "answerable": len(answerable), "unanswerable": len(unanswerable),
        "answerable_outcomes": {o: share(answerable, "outcome", o) for o in
                                ("accepted", "rejected", "abstained_model", "abstained_threshold", "generator_error")},
        "accepted_citing_a_relevant_passage": round(sum(bool(r["cites_relevant"]) for r in accepted) / len(accepted), 4) if accepted else None,
        "unanswerable_outcomes": {o: share(unanswerable, "outcome", o) for o in
                                  ("abstained_threshold", "abstained_model", "accepted", "rejected", "generator_error")},
        "unanswerable_abstained_total": round(sum(r["outcome"].startswith("abstained") for r in unanswerable) / len(unanswerable), 4) if unanswerable else None,
        "seconds_median": statistics.median(seconds) if seconds else None, "seconds_max": max(seconds) if seconds else None,
    }


def report_text(meta: dict) -> str:
    s = meta["summary"]
    rows = meta["results"]
    accepted = [r for r in rows if r["outcome"] == "accepted"]
    rejected = [r for r in rows if r["outcome"] == "rejected"]
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    return "\n".join([
        "# RAG Generation Evaluation (local Ollama model)", "",
        f"Generated (UTC): {meta['generated_at_utc']}. Model `{meta['model']}` (Ollama {meta['ollama_version']}), "
        f"options {meta['options']}. Index: {meta['index_chunks']} chunks; retrieval threshold {meta['min_relevance']}; "
        f"grounding threshold {meta['min_support']}.", "",
        "Automatic checks are lexical: an accepted answer cites retrieved passages, and its numbers and most of its "
        "words occur in them. That does not prove the answer is correct; accepted answers are listed below for review.", "",
        f"## Answerable questions ({s['answerable']})", "",
        "| Outcome | Share |", "|---|---|",
        *[f"| {k.replace('_', ' ')} | {pct(v)} |" for k, v in s["answerable_outcomes"].items()], "",
        f"Accepted answers citing at least one expected relevant passage: {pct(s['accepted_citing_a_relevant_passage'])}", "",
        f"## Unanswerable questions ({s['unanswerable']})", "",
        "| Outcome | Share |", "|---|---|",
        *[f"| {k.replace('_', ' ')} | {pct(v)} |" for k, v in s["unanswerable_outcomes"].items()], "",
        f"Abstained in total: {pct(s['unanswerable_abstained_total'])}", "",
        f"Latency: median {s['seconds_median']} s, max {s['seconds_max']} s per generated answer (CPU).", "",
        "## Accepted generated answers (for manual review)", "",
        "| Id | Question | Answer | Cites relevant passage |", "|---|---|---|---|",
        *[f"| {r['id']} | {r['question']} | {r['final_answer'].replace('|', '/')} | {r['cites_relevant']} |" for r in accepted], "",
        "## Rejected generated answers (extractive answer used instead)", "",
        "| Id | Model answer | First problem |", "|---|---|---|",
        *[f"| {r['id']} | {(r['raw_answer'] or '').replace('|', '/')[:200]} | {(r['grounding_problems'] or [''])[0][:120]} |" for r in rejected], "",
        "## Limitations", "",
        "- 32 developer-written questions about this project's own documents; shares carry wide uncertainty.",
        "- Deterministic decoding (temperature 0, fixed seed) on one machine; other hardware or Ollama versions may differ.",
        "- The model is Qwen2.5-3B (Qwen Research License, non-commercial use only).", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing report")
    args = parser.parse_args()
    settings = pipeline.load_rag_settings()
    root, o = pipeline.PROJECT_ROOT, settings["ollama"]
    out_dir = root / settings["reports_dir"]
    outputs = [out_dir / "generation_eval.json", out_dir / "generation_eval.md"]
    if any(p.exists() for p in outputs) and not args.force:
        print(f"Report already exists in {out_dir}. Nothing was written. Use --force to overwrite.")
        return 1
    status = ans.ollama_status(o["url"], o["model"])
    if not status["available"]:
        print(f"Failed: {status['reason']}")
        return 1
    try:
        store, manifest, embeddings = pipeline.open_store(settings)
    except (RuntimeError, idx.RAGIndexError) as error:
        print(f"Failed: {error}")
        return 1
    questions = json.loads((root / settings["eval_set"]).read_text(encoding="utf-8"))["questions"]
    generator = ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"], settings["max_answer_sentences"])
    print(f"Generating answers for {len(questions)} questions with {o['model']} (this takes a while on CPU)")
    rows = evaluate(questions, store, embeddings, settings, generator)
    meta = {"generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": o["model"],
            "ollama_version": status["version"], "options": o["options"], "min_relevance": settings["min_relevance"],
            "min_support": o["min_support"], "index_chunks": manifest["chunks"],
            "embedding_model": manifest["embedding_model"], "summary": summarize(rows), "results": rows,
            "versions": pipeline.versions()}
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in ((outputs[0], json.dumps(meta, indent=2, ensure_ascii=False) + "\n"), (outputs[1], report_text(meta))):
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    print(json.dumps(meta["summary"], indent=2))
    print(f"Saved {outputs[0]} and {outputs[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
