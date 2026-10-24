"""Retrieval evaluation against a labelled question set (docs/decisions.md, D14).

A retrieved chunk is relevant to a question when it comes from one of the
question's expected sources and contains the expected phrase (case- and
whitespace-insensitive). Questions marked "answerable": false have no relevant
chunk; they measure whether the relevance threshold correctly abstains.
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: dict, targets: list[dict]) -> bool:
    text = normalize(chunk["text"])
    return any(chunk["source"] == t["source"] and normalize(t["contains"]) in text for t in targets)


def validate_questions(questions: list[dict], chunks: list[dict]) -> list[str]:
    """Problems with the question set: every answerable target phrase must occur in some indexed chunk."""
    problems, ids = [], set()
    for q in questions:
        if q["id"] in ids:
            problems.append(f"duplicate id {q['id']}")
        ids.add(q["id"])
        if q.get("answerable", True):
            if not q.get("relevant"):
                problems.append(f"{q['id']}: answerable question without relevant targets")
            elif not any(is_relevant(c, q["relevant"]) for c in chunks):
                problems.append(f"{q['id']}: no indexed chunk contains its target phrase")
    return problems


def first_relevant_rank(hits: list[dict], targets: list[dict]) -> int | None:
    for hit in hits:
        if is_relevant(hit, targets):
            return hit["rank"]
    return None


def summarize(results: list[dict], ks: list[int], min_relevance: float) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]
    summary = {"questions": len(results), "answerable": len(answerable), "unanswerable": len(unanswerable)}
    for k in ks:
        summary[f"hit@{k}"] = float(np.mean([r["rank"] is not None and r["rank"] <= k for r in answerable])) if answerable else None
    summary["mrr"] = float(np.mean([1 / r["rank"] if r["rank"] else 0.0 for r in answerable])) if answerable else None
    if all(r["top_score"] is not None for r in results):
        summary["answered_when_answerable"] = float(np.mean([r["top_score"] >= min_relevance for r in answerable])) if answerable else None
        summary["abstained_when_unanswerable"] = float(np.mean([r["top_score"] < min_relevance for r in unanswerable])) if unanswerable else None
    return summary


def evaluate(search, questions: list[dict], ks: list[int], min_relevance: float) -> dict:
    """`search(question, k) -> list of hits` (each with rank, score, source, text)."""
    max_k = max(ks)
    results = []
    for q in questions:
        hits = search(q["question"], max_k)
        answerable = q.get("answerable", True)
        results.append({"id": q["id"], "question": q["question"], "answerable": answerable,
                        "rank": first_relevant_rank(hits, q["relevant"]) if answerable else None,
                        "top_score": hits[0]["score"] if hits else None,
                        "top_source": hits[0]["source"] if hits else None})
    return {"summary": summarize(results, ks, min_relevance), "results": results}


def tfidf_search(chunks: list[dict]):
    """A lexical baseline over the same chunks (TF-IDF cosine), for comparison only."""
    vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])

    def search(question: str, k: int) -> list[dict]:
        scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")[:k]
        return [{"rank": i + 1, "score": float(scores[j]), **chunks[j]} for i, j in enumerate(order)]
    return search
