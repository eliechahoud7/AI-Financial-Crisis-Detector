"""Tests for wider retrieval with a cross-encoder reranker (docs/decisions.md, D31).

Unit tests use a deterministic fake reranker, so they need no model. One test
uses the real cross-encoder and is skipped if it is not in the local cache.
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from langchain_core.documents import Document

from fincrisis.qa import rag_regression as rr
from fincrisis.rag import answer as ans
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline
from fincrisis.rag.retrieval import Retriever, rerank_text
from tests.test_rag import BagOfWordsEmbedding, model_cached

RERANKER = pipeline.load_rag_settings()["retrieval"]["rerank_model"]

DOCS = [
    ("docs/a.md", "A > Liquidity", "Liquidity risk means a firm cannot meet short-term obligations."),
    ("docs/a.md", "A > Leverage", "Leverage measures debt relative to equity in a firm."),
    ("reports/test.md", "Held-out test > Paired differences", "xgboost minus rule: -0.0076 on the firm sample."),
    ("reports/cv.md", "Walk-forward validation > Paired differences", "xgboost minus rule: -0.0823 on the firm sample."),
    ("docs/b.md", "B > Weather", "Rain is expected in the afternoon."),
]


def make_store():
    documents = [Document(page_content=text, metadata={"source": source, "section": section, "chunk_id": f"{source}#{i}"})
                 for i, (source, section, text) in enumerate(DOCS, start=1)]
    return idx.FaissStore.from_documents(documents, BagOfWordsEmbedding())


class Recorder:
    """A fake reranker: scores by a preferred phrase, and records what it was asked to read."""

    def __init__(self, prefer: str):
        self.prefer, self.calls = prefer, []

    def __call__(self, question, passages):
        self.calls.append((question, list(passages)))
        return [1.0 if self.prefer in p else 0.0 for p in passages]


class TestRerankMode(unittest.TestCase):
    def test_candidates_are_the_dense_top_and_are_re_ordered(self):
        store = make_store()
        reranker = Recorder("Held-out test")
        retriever = Retriever(store, "rerank", reranker=reranker, candidates=3)
        hits = retriever.search("xgboost minus rule firm", 2)
        dense_top = [store.documents[i].metadata["chunk_id"] for i in
                     sorted(range(len(DOCS)), key=lambda i: -store.cosine_scores("xgboost minus rule firm")[i])[:3]]
        self.assertEqual(len(reranker.calls[0][1]), 3)  # it reads exactly the dense candidates
        self.assertEqual({p.split("\n", 1)[1] for p in reranker.calls[0][1]},
                         {d.page_content for d in store.documents if d.metadata["chunk_id"] in dense_top})
        self.assertEqual(hits[0]["source"], "reports/test.md")  # the reranker's choice comes first
        self.assertEqual([h["rank"] for h in hits], [1, 2])
        dense = store.cosine_scores("xgboost minus rule firm")
        for hit in hits:  # the score stays the dense cosine, so the threshold keeps its meaning
            i = next(k for k, d in enumerate(store.documents) if d.metadata["chunk_id"] == hit["chunk_id"])
            self.assertAlmostEqual(hit["score"], float(dense[i]))

    def test_the_reranker_reads_the_source_and_section(self):
        store = make_store()
        reranker = Recorder("Walk-forward validation")
        Retriever(store, "rerank", reranker=reranker, candidates=5).search("xgboost minus rule", 1)
        read = reranker.calls[0][1]
        self.assertIn("reports/test.md - Held-out test > Paired differences\nxgboost minus rule: -0.0076 on the firm sample.",
                      read)
        self.assertEqual(rerank_text(store.documents[0]), f"docs/a.md - A > Liquidity\n{DOCS[0][2]}")

    def test_more_results_than_candidates_reads_k_passages(self):
        reranker = Recorder("nothing")
        hits = Retriever(make_store(), "rerank", reranker=reranker, candidates=2).search("firm", 4)
        self.assertEqual((len(reranker.calls[0][1]), len(hits)), (4, 4))

    def test_invalid_settings_are_refused(self):
        with self.assertRaises(ValueError):
            Retriever(make_store(), "rerank")
        with self.assertRaises(ValueError):
            Retriever(make_store(), "rerank", reranker=Recorder("x"), candidates=0)

    def test_the_relevance_gate_is_still_the_best_dense_cosine(self):
        store = make_store()
        retriever = Retriever(store, "rerank", reranker=Recorder("Weather"), candidates=5)
        question = "liquidity obligations firm"
        best = float(store.cosine_scores(question).max())
        refused = ans.answer(question, store, store.embeddings, 2, best + 0.01, 1, retriever=retriever)
        self.assertEqual(refused["status"], "insufficient_evidence")  # a high rerank score cannot open the gate
        answered = ans.answer(question, store, store.embeddings, 2, best - 0.01, 1, retriever=retriever)
        self.assertEqual((answered["status"], answered["evidence"][0]["source"]), ("answered", "docs/b.md"))


class TestPipeline(unittest.TestCase):
    SETTINGS = {"retrieval": {"mode": "rerank", "rrf_k": 60, "rerank_model": "some/model", "rerank_candidates": 7}}

    def test_make_retriever_loads_the_configured_reranker(self):
        with unittest.mock.patch.object(pipeline, "make_reranker", return_value=Recorder("x")) as loader:
            retriever = pipeline.make_retriever(make_store(), self.SETTINGS)
        loader.assert_called_once_with("some/model")
        self.assertEqual((retriever.mode, retriever.candidates), ("rerank", 7))
        with unittest.mock.patch.object(pipeline, "make_reranker", side_effect=AssertionError("loaded")):
            self.assertEqual(pipeline.make_retriever(make_store(), self.SETTINGS, "hybrid").mode, "hybrid")

    def test_a_missing_reranker_model_is_a_clear_error(self):
        with unittest.mock.patch("sentence_transformers.CrossEncoder", side_effect=OSError("not cached")):
            with self.assertRaisesRegex(RuntimeError, "not in the local cache.*--allow-download"):
                pipeline.make_reranker("some/model")
            with self.assertRaises(OSError):  # with downloads allowed, the original error is not hidden
                pipeline.make_reranker("some/model", allow_download=True)

    def test_rerank_is_evaluated_only_when_configured(self):
        self.assertEqual(pipeline.retrieval_modes(self.SETTINGS), ("dense", "lexical", "hybrid", "rerank"))
        self.assertEqual(pipeline.retrieval_modes({"retrieval": {"mode": "dense"}}), ("dense", "lexical", "hybrid"))


class TestRegressionBaseline(unittest.TestCase):
    def test_a_configured_mode_without_a_baseline_must_be_reviewed(self):
        mode = {"summary": {"mrr": 0.5}, "ranks": {"q1": 1}}
        baseline = {"modes": {"dense": mode}}
        checks = rr.compare_retrieval({"dense": mode, "rerank": mode}, baseline, "rerank")
        [check] = [c for c in checks if "rerank" in c.group]
        self.assertFalse(check.ok)
        self.assertIn("no baseline", check.detail)
        reported = rr.compare_retrieval({"dense": mode, "rerank": mode}, baseline, "dense")
        self.assertTrue(all(c.ok for c in reported))  # a mode that is only reported never fails the check


@unittest.skipUnless(model_cached(RERANKER), "reranker model not in the local cache")
class TestRealReranker(unittest.TestCase):
    def test_relevant_passage_scores_higher(self):
        rerank = pipeline.make_reranker(RERANKER)
        relevant, unrelated = rerank("How many people live in Berlin?", [
            "Berlin has a population of 3,520,031 registered inhabitants.", "Rain is expected in the afternoon."])
        self.assertGreater(relevant, unrelated)


if __name__ == "__main__":
    unittest.main()
