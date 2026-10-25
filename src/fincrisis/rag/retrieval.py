"""Dense, lexical, and hybrid retrieval over the same chunks (docs/decisions.md, D16).

- dense: cosine similarity of sentence embeddings (the FAISS index).
- lexical: TF-IDF cosine over chunk text plus section heading, fitted locally
  when the index is loaded; nothing extra is stored.
- hybrid: reciprocal rank fusion (RRF) of the dense and lexical rankings,
  score = sum over rankings of 1 / (rrf_k + rank).
- rerank (D31): the top `candidates` of the dense ranking, re-ordered by a
  cross-encoder that reads the question and each passage (with its source
  and section) together.

Every hit carries its dense cosine as "score", whatever the ranking method,
so the relevance threshold for abstaining keeps the same meaning (D14).
"""

from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from fincrisis.rag.index import mask_digests

MODES = ("dense", "lexical", "hybrid", "rerank")


def rerank_text(document, mask: bool = False) -> str:
    """What the reranker reads: the citation location, then the passage. Without it, rows of the held-out test and
    of the validation report (same tables, different numbers) cannot be told apart. With `mask`, long hex digests
    are replaced by a placeholder (D33)."""
    text = mask_digests(document.page_content) if mask else document.page_content
    return f"{document.metadata.get('source', '')} - {document.metadata.get('section', '')}\n{text}"


class Retriever:
    def __init__(self, store, mode: str = "dense", rrf_k: int = 60, include_section: bool = True,
                 reranker: Callable[[str, list[str]], list[float]] | None = None, candidates: int = 20,
                 mask: bool = False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "rerank" and (reranker is None or candidates < 1):
            raise ValueError("rerank mode needs a reranker and at least one candidate")
        self.store, self.mode, self.rrf_k = store, mode, rrf_k
        self.reranker, self.candidates, self.mask = reranker, candidates, mask
        self.sentence_scorer = None  # set by pipeline.make_retriever when the reranker also picks sentences (D32)
        self.relevant_continuation = False  # D32 experiment: continue only with sentences the reranker scores above 0
        texts = [(f"{d.metadata.get('section', '')}\n" if include_section else "") + d.page_content
                 for d in store.documents]
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(texts)

    def lexical_scores(self, question: str) -> np.ndarray:
        return (self.matrix @ self.vectorizer.transform([question]).T).toarray().ravel()

    def search(self, question: str, k: int) -> list[dict]:
        dense = self.store.cosine_scores(question)
        if self.mode == "dense":
            ranking = np.argsort(-dense, kind="stable")
        elif self.mode == "lexical":
            ranking = np.argsort(-self.lexical_scores(question), kind="stable")
        elif self.mode == "rerank":
            pool = np.argsort(-dense, kind="stable")[:max(k, self.candidates)]
            relevance = np.asarray(self.reranker(question, [rerank_text(self.store.documents[i], self.mask) for i in pool]))
            ranking = pool[np.argsort(-relevance, kind="stable")]
        else:
            fused = np.zeros(len(dense))
            for scores in (dense, self.lexical_scores(question)):
                ranks = np.empty(len(scores), dtype=int)
                ranks[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
                fused += 1.0 / (self.rrf_k + ranks)
            ranking = np.argsort(-fused, kind="stable")
        hits = []
        for rank, i in enumerate(ranking[:k], start=1):
            doc = self.store.documents[i]
            hits.append({"rank": rank, "score": float(dense[i]), "source": doc.metadata["source"],
                         "section": doc.metadata["section"], "chunk_id": doc.metadata["chunk_id"],
                         "text": doc.page_content})
        return hits

    def max_dense_score(self, question: str) -> float:
        return float(self.store.cosine_scores(question).max())
