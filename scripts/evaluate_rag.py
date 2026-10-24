"""Evaluate retrieval quality on the labelled question set (docs/decisions.md, D14).

Usage:
    python scripts/evaluate_rag.py            # stop if a report exists
    python scripts/evaluate_rag.py --force

Compares the dense retriever (the RAG index) with a TF-IDF baseline over the
same chunks. Reports hit@k, mean reciprocal rank, and how often the fixed
relevance threshold answers answerable questions and abstains on the
unanswerable ones. Writes data/reports/rag/retrieval_eval.json and .md.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from fincrisis.rag import answer as ans
from fincrisis.rag import evaluation as ev
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline


def chunk_dicts(store) -> list[dict]:
    return [{"source": d.metadata["source"], "section": d.metadata["section"], "chunk_id": d.metadata["chunk_id"],
             "text": d.page_content} for d in store.documents]


def report_text(result: dict) -> str:
    rows = []
    for name, r in result["retrievers"].items():
        s = r["summary"]
        rows.append(f"| {name} | " + " | ".join(f"{s[f'hit@{k}']:.3f}" for k in result["eval_k"])
                    + f" | {s['mrr']:.3f} | {s.get('answered_when_answerable', float('nan')):.3f} | "
                    f"{s.get('abstained_when_unanswerable', float('nan')):.3f} |")
    misses = [r for r in result["retrievers"]["dense"]["results"] if r["answerable"] and (r["rank"] is None or r["rank"] > 1)]
    miss_rows = [f"| {r['id']} | {r['question']} | {r['rank'] or '>' + str(max(result['eval_k']))} | {r['top_source']} |" for r in misses]
    wrong_abstain = [r for r in result["retrievers"]["dense"]["results"]
                     if not r["answerable"] and r["top_score"] >= result["min_relevance"]]
    return "\n".join([
        "# RAG Retrieval Evaluation", "",
        f"Generated (UTC): {result['generated_at_utc']}. Index: {result['index']['chunks']} chunks from "
        f"{len(result['index']['sources'])} documents; embedding model `{result['index']['embedding_model']}`.", "",
        f"Question set: {result['questions']} questions ({result['answerable']} answerable, "
        f"{result['questions'] - result['answerable']} unanswerable). A hit is a chunk from the expected source "
        "containing the expected phrase. The relevance threshold "
        f"({result['min_relevance']}) was fixed before evaluation, not tuned on these questions.", "",
        "| Retriever | " + " | ".join(f"hit@{k}" for k in result["eval_k"]) + " | MRR | Answered (answerable) | Abstained (unanswerable) |",
        "|---|" + "---|" * (len(result["eval_k"]) + 3),
        *rows, "",
        "TF-IDF scores are not cosine similarities of the same kind, so its answer/abstain rates use the same "
        "threshold only for reference.", "",
        "## Dense retriever: questions not answered by the first passage", "",
        "| Id | Question | Rank of first relevant | Top source |", "|---|---|---|---|", *miss_rows, "",
        f"Unanswerable questions that passed the threshold: {[r['id'] for r in wrong_abstain] or 'none'}", "",
        "## Limitations", "",
        "- The question set is small and written by the developer from the project's own documents, so these "
        "numbers measure this corpus only and carry wide uncertainty.",
        "- Relevance is judged by an expected phrase, which can miss other chunks that also answer the question.", "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite an existing report")
    args = parser.parse_args()
    settings = pipeline.load_rag_settings()
    root = pipeline.PROJECT_ROOT
    out_dir = root / settings["reports_dir"]
    outputs = [out_dir / "retrieval_eval.json", out_dir / "retrieval_eval.md"]
    if any(p.exists() for p in outputs) and not args.force:
        print(f"Report already exists in {out_dir}. Nothing was written. Use --force to overwrite.")
        return 1
    try:
        store, manifest, _ = pipeline.open_store(settings)
    except (RuntimeError, idx.RAGIndexError) as error:
        print(f"Failed: {error}")
        return 1
    questions = json.loads((root / settings["eval_set"]).read_text(encoding="utf-8"))["questions"]
    chunks = chunk_dicts(store)
    problems = ev.validate_questions(questions, chunks)
    if problems:
        print("Question set is invalid for this index:\n  " + "\n  ".join(problems))
        return 1
    ks, threshold = settings["eval_k"], settings["min_relevance"]
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": {k: manifest[k] for k in ("embedding_model", "embedding_model_revision", "chunks", "sources", "chunk_size", "chunk_overlap")},
        "eval_set": settings["eval_set"], "questions": len(questions),
        "answerable": sum(q.get("answerable", True) for q in questions), "eval_k": ks, "min_relevance": threshold,
        "retrievers": {"dense": ev.evaluate(lambda q, k: ans.retrieve(store, q, k), questions, ks, threshold),
                       "tfidf_baseline": ev.evaluate(ev.tfidf_search(chunks), questions, ks, threshold)},
        "versions": pipeline.versions(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, text in ((outputs[0], json.dumps(result, indent=2, ensure_ascii=False) + "\n"), (outputs[1], report_text(result))):
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
    for name, r in result["retrievers"].items():
        s = r["summary"]
        print(f"{name:15} " + "  ".join(f"hit@{k} {s[f'hit@{k}']:.3f}" for k in ks) + f"  MRR {s['mrr']:.3f}  "
              f"answered {s['answered_when_answerable']:.3f}  abstained {s['abstained_when_unanswerable']:.3f}")
    print(f"Saved {outputs[0]} and {outputs[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
