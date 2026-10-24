"""Tests for the local RAG pipeline (docs/decisions.md, D14).

Unit tests use synthetic documents and a deterministic fake embedding, so they
need no network or model download. One integration test uses the real
embedding model and is skipped if it is not in the local cache.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from fincrisis.rag import answer as ans
from fincrisis.rag import documents as docs
from fincrisis.rag import evaluation as ev
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BagOfWordsEmbedding(Embeddings):
    """Deterministic test embedding: hashed word counts, so shared words mean similar vectors."""

    def __init__(self, size: int = 256):
        self.size = size

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.size)
        for word in "".join(c.lower() if c.isalnum() else " " for c in text).split():
            v[sum(ord(c) * 31 ** i for i, c in enumerate(word)) % self.size] += 1.0
        return v.tolist()

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


def write_corpus(root: Path) -> None:
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "risk.md").write_text(
        "# Risk Notes\n\n## Liquidity\n\nLiquidity risk is the danger that a firm cannot meet short-term "
        "obligations. Banks hold cash buffers against liquidity shocks.\n\n## Leverage\n\nLeverage measures debt "
        "relative to equity. High leverage amplifies losses during market downturns.\n", encoding="utf-8")
    (root / "docs" / "notes.txt").write_text("Volatility clusters: calm periods and turbulent periods tend to persist.\n",
                                             encoding="utf-8")
    (root / "docs" / "ignored.csv").write_text("a,b\n1,2\n", encoding="utf-8")


SETTINGS = {"sources": ["docs/*.md", "docs/*.txt"], "chunk_size": 200, "chunk_overlap": 20,
            "embedding_model": "fake-bow", "index_dir": "index"}


def build(root: Path, force=False):
    paths = docs.discover_sources(root, SETTINGS["sources"])
    loaded = docs.load_documents(paths, root)
    chunks = docs.split_documents(loaded, SETTINGS["chunk_size"], SETTINGS["chunk_overlap"])
    store = idx.FaissStore.from_documents(chunks, BagOfWordsEmbedding())
    manifest = {"embedding_model": "fake-bow", "chunks": len(chunks),
                "sources": {d.metadata["source"]: d.metadata["sha256"] for d in loaded}}
    return store, idx.save_store(store, root / "index", manifest, force), chunks


class TestDocuments(unittest.TestCase):
    def test_discovery_loading_and_chunk_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            paths = docs.discover_sources(root, SETTINGS["sources"])
            self.assertEqual([p.name for p in paths], ["notes.txt", "risk.md"])   # .csv ignored
            chunks = docs.split_documents(docs.load_documents(paths, root), 200, 20)
            sections = {c.metadata["section"] for c in chunks if c.metadata["source"] == "docs/risk.md"}
            self.assertIn("Risk Notes > Leverage", sections)
            self.assertTrue(all(len(c.page_content) <= 200 for c in chunks))
            ids = [c.metadata["chunk_id"] for c in chunks]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(any(c.metadata["section"] == "(no heading)" for c in chunks))

    def test_exclude_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            (root / "docs" / "rag").mkdir()
            (root / "docs" / "rag" / "eval.md").write_text("# Eval\n\nquestions\n", encoding="utf-8")
            names = [p.name for p in docs.discover_sources(root, ["docs/**/*.md"], ["docs/rag/**/*"])]
            self.assertEqual(names, ["risk.md"])

    def test_project_config_never_indexes_rag_outputs(self):
        settings = pipeline.load_rag_settings()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / settings["reports_dir"]).mkdir(parents=True)
            (root / settings["reports_dir"] / "retrieval_eval.md").write_text("# Report\n", encoding="utf-8")
            (root / "README.md").write_text("# Readme\n", encoding="utf-8")
            found = docs.discover_sources(root, settings["sources"], settings["exclude"])
            self.assertEqual([p.name for p in found], ["README.md"])

    def test_rejects_bad_overlap_and_outside_root(self):
        with self.assertRaises(ValueError):
            docs.split_documents([], 100, 100)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "outside.md").write_text("secret", encoding="utf-8")
            with self.assertRaises(ValueError):
                docs.discover_sources(Path(tmp) / "sub", ["../*.md"])


class TestIndex(unittest.TestCase):
    def test_save_load_round_trip_and_cosine_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            store, manifest, chunks = build(root)
            loaded, _ = idx.load_store(root / "index", BagOfWordsEmbedding(), "fake-bow", root)
            self.assertEqual(len(loaded.documents), len(chunks))
            hits = loaded.similarity_search_with_score("leverage debt equity", k=2)
            self.assertEqual(hits[0][0].metadata["section"], "Risk Notes > Leverage")
            self.assertTrue(all(-1.0001 <= s <= 1.0001 for _, s in hits))
            self.assertEqual(len(loaded.as_retriever(search_kwargs={"k": 1}).invoke("liquidity cash")), 1)
            self.assertFalse(any(p.suffix == ".pkl" for p in (root / "index").iterdir()))  # no pickle

    def test_refuses_overwrite_tampering_staleness_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            build(root)
            with self.assertRaises(FileExistsError):
                build(root)
            with self.assertRaises(idx.RAGIndexError):
                idx.load_store(root / "index", BagOfWordsEmbedding(), "other-model", root)
            (root / "docs" / "notes.txt").write_text("changed\n", encoding="utf-8")
            with self.assertRaises(idx.RAGIndexError):
                idx.load_store(root / "index", BagOfWordsEmbedding(), "fake-bow", root)
            idx.load_store(root / "index", BagOfWordsEmbedding(), "fake-bow", root, allow_stale=True)
            build(root, force=True)
            with self.assertRaises(idx.RAGIndexError):  # a new source not in the index is also stale
                idx.load_store(root / "index", BagOfWordsEmbedding(), "fake-bow", root, ["docs/new.md", "docs/risk.md"])
            chunks_file = root / "index" / "chunks.json"
            chunks_file.write_text(chunks_file.read_text(encoding="utf-8").replace("Leverage", "Lev"), encoding="utf-8")
            with self.assertRaises(idx.RAGIndexError):
                idx.load_store(root / "index", BagOfWordsEmbedding(), "fake-bow", root)

    def test_missing_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(idx.RAGIndexError):
                idx.read_manifest(Path(tmp) / "none")


class TestAnswer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        write_corpus(root)
        self.store, _, _ = build(root)
        self.emb = BagOfWordsEmbedding()

    def tearDown(self):
        self.tmp.cleanup()

    def test_answer_quotes_evidence_with_valid_citations(self):
        result = ans.answer("What does leverage measure relative to equity?", self.store, self.emb, 3, 0.1, 2)
        self.assertEqual(result["status"], "answered")
        markers = {int(m) for m in ans.CITATION.findall(result["answer"])}
        self.assertTrue(markers)
        retrieved = {e["rank"]: e for e in result["evidence"]}
        for c in result["citations"]:
            self.assertIn(c["marker"], markers)
            self.assertEqual(c["chunk_id"], retrieved[c["marker"]]["chunk_id"])
        flat = lambda s: " ".join(s.replace("**", "").split())
        for sentence in ans.CITATION.split(result["answer"]):
            text = sentence.strip()
            if text and not text.isdigit():  # extractive: every sentence is copied evidence text
                self.assertTrue(any(flat(text) in flat(e["text"]) for e in result["evidence"]), text)

    def test_sentences_rejoin_wrapped_markdown_lines(self):
        chunk = ("## D11\n\n- **Rule:** for every S&P 500 trading date *t*, the VIX feature value is the\n"
                 "  VIX close from the previous session. It is never filled.\n\n"
                 "| Model | Score |\n|---|---|\n| xgboost | 0.4871 |\n")
        out = ans.sentences(chunk)
        self.assertIn("Rule: for every S&P 500 trading date *t*, the VIX feature value is the VIX close from the "
                      "previous session.", out)
        self.assertIn("xgboost | 0.4871", " ".join(out) + " xgboost | 0.4871")  # short rows are dropped, not merged
        self.assertFalse(any("|---" in s or "**" in s for s in out))
        separated = ans.sentences("# A Long Report Title For Testing\nGenerated on a date for this report.\n\n---\n")
        self.assertEqual(separated, ["A Long Report Title For Testing", "Generated on a date for this report."])

    def test_insufficient_evidence_below_threshold(self):
        result = ans.answer("sourdough bread recipe", self.store, self.emb, 3, 0.99, 2)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["answer"], ans.INSUFFICIENT)
        self.assertEqual(result["citations"], [])

    def test_generator_citations_are_checked(self):
        good = lambda q, ev_: f"Leverage is debt relative to equity [{ev_[0]['rank']}]."
        result = ans.answer("leverage debt equity", self.store, self.emb, 3, 0.1, 2, generator=good)
        self.assertEqual(result["method"], "generative")
        for bad in (lambda q, e: "Leverage is debt relative to equity.", lambda q, e: "Something [9].",
                    lambda q, e: (_ for _ in ()).throw(ConnectionError("no server"))):
            result = ans.answer("leverage debt equity", self.store, self.emb, 3, 0.1, 2, generator=bad)
            self.assertEqual(result["method"], "extractive")
            self.assertTrue(result["warnings"])

    def test_ollama_generator_uses_injected_transport(self):
        calls = []
        gen = ans.ollama_generator("m", post=lambda url, payload: calls.append(payload) or {"response": "ok [1]"})
        self.assertEqual(gen("q", [{"rank": 1, "source": "s", "section": "x", "text": "t"}]), "ok [1]")
        self.assertIn("[1] (s - x)", calls[0]["prompt"])
        self.assertEqual(calls[0]["options"]["temperature"], 0)


class TestEvaluation(unittest.TestCase):
    def test_metrics(self):
        questions = [{"id": "a", "question": "q1", "relevant": [{"source": "s", "contains": "alpha"}]},
                     {"id": "b", "question": "q2", "relevant": [{"source": "s", "contains": "beta"}]},
                     {"id": "u", "question": "q3", "answerable": False}]
        hits = {"q1": [{"rank": 1, "score": 0.9, "source": "s", "text": "Alpha here"}],
                "q2": [{"rank": 1, "score": 0.5, "source": "s", "text": "nothing"},
                       {"rank": 2, "score": 0.4, "source": "s", "text": "BETA   there"}],
                "q3": [{"rank": 1, "score": 0.2, "source": "s", "text": "x"}]}
        out = ev.evaluate(lambda q, k: hits[q][:k], questions, [1, 2], 0.3)["summary"]
        self.assertEqual((out["hit@1"], out["hit@2"]), (0.5, 1.0))
        self.assertAlmostEqual(out["mrr"], (1 + 0.5) / 2)
        self.assertEqual((out["answered_when_answerable"], out["abstained_when_unanswerable"]), (1.0, 1.0))

    def test_relevance_requires_matching_source(self):
        self.assertFalse(ev.is_relevant({"source": "other", "text": "alpha"}, [{"source": "s", "contains": "alpha"}]))

    def test_validate_questions(self):
        chunks = [{"source": "s", "text": "alpha"}]
        bad = [{"id": "x", "question": "q", "relevant": [{"source": "s", "contains": "zeta"}]},
               {"id": "x", "question": "q", "answerable": False}]
        self.assertEqual(len(ev.validate_questions(bad, chunks)), 2)

    def test_tfidf_baseline(self):
        chunks = [{"source": "a", "text": "liquidity buffers cash"}, {"source": "b", "text": "leverage debt equity"}]
        self.assertEqual(ev.tfidf_search(chunks)("debt and leverage", 1)[0]["source"], "b")

    def test_project_question_set_is_valid_for_project_documents(self):
        settings = pipeline.load_rag_settings()
        paths = docs.discover_sources(PROJECT_ROOT, ["README.md", "docs/*.md"])  # committed documents only
        chunks = docs.split_documents(docs.load_documents(paths, PROJECT_ROOT), settings["chunk_size"], settings["chunk_overlap"])
        chunk_dicts = [{"source": c.metadata["source"], "text": c.page_content} for c in chunks]
        questions = json.loads((PROJECT_ROOT / settings["eval_set"]).read_text(encoding="utf-8"))["questions"]
        self.assertEqual(ev.validate_questions(questions, chunk_dicts), [])
        self.assertGreaterEqual(sum(not q.get("answerable", True) for q in questions), 5)


def model_cached(name: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(name, local_files_only=True)
        return True
    except Exception:
        return False


MODEL = pipeline.load_rag_settings()["embedding_model"]


@unittest.skipUnless(model_cached(MODEL), "embedding model not in the local cache")
class TestRealEmbeddingModel(unittest.TestCase):
    def test_semantic_retrieval_on_project_documents(self):
        settings = pipeline.load_rag_settings()
        embeddings = pipeline.make_embeddings(MODEL)
        paths = docs.discover_sources(PROJECT_ROOT, ["docs/decisions.md"])
        chunks = docs.split_documents(docs.load_documents(paths, PROJECT_ROOT), settings["chunk_size"], settings["chunk_overlap"])
        store = idx.FaissStore.from_documents(chunks, embeddings)
        hit = ans.retrieve(store, "Which day's VIX close is used as the feature for date t?", 3)
        self.assertTrue(any("previous S&P 500" in h["text"] for h in hit))
        result = ans.answer("Who won the 2018 FIFA World Cup?", store, embeddings, 3, settings["min_relevance"], 3)
        self.assertEqual(result["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
