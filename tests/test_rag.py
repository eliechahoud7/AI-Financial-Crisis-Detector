"""Tests for the local RAG pipeline (docs/decisions.md, D14).

Unit tests use synthetic documents and a deterministic fake embedding, so they
need no network or model download. One integration test uses the real
embedding model and is skipped if it is not in the local cache.
"""

import json
import tempfile
import unittest
import unittest.mock
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

    def test_generated_answer_accepted_only_when_grounded(self):
        info = {"model": "fake", "seconds": 0.1}
        good = lambda q, ev_: (f"Leverage measures debt relative to equity [{ev_[0]['rank']}].", info)
        result = ans.answer("leverage debt equity", self.store, self.emb, 3, 0.1, 2, generator=good)
        self.assertEqual((result["method"], result["status"]), ("generative", "answered"))
        self.assertEqual(result["generation"]["grounding_problems"], [])
        self.assertTrue(result["citations"])
        bad_answers = (
            lambda q, e: ("Leverage measures debt relative to equity.", info),               # no citation
            lambda q, e: ("Leverage measures debt [9].", info),                                # unretrieved passage
            lambda q, e: (f"Leverage rose 37% in 2019 [{e[0]['rank']}].", info),              # invented numbers
            lambda q, e: (f"Quantum chromodynamics governs gluons [{e[0]['rank']}].", info),   # unsupported words
            lambda q, e: (_ for _ in ()).throw(ConnectionError("no server")),                 # server down
        )
        for bad in bad_answers:
            result = ans.answer("leverage debt equity", self.store, self.emb, 3, 0.1, 2, generator=bad)
            self.assertEqual(result["method"], "extractive")
            self.assertEqual(result["status"], "answered")
            self.assertTrue(result["warnings"])

    def test_model_abstention_becomes_insufficient_evidence(self):
        abstain = lambda q, e: (ans.INSUFFICIENT, {"model": "fake"})
        result = ans.answer("leverage debt equity", self.store, self.emb, 3, 0.1, 2, generator=abstain)
        self.assertEqual((result["status"], result["method"], result["citations"]), ("insufficient_evidence", "generative", []))
        self.assertTrue(ans.is_abstention("Insufficient evidence."))
        self.assertFalse(ans.is_abstention("Leverage is debt over equity [1]."))

    def test_threshold_abstention_skips_the_model(self):
        calls = []
        result = ans.answer("sourdough", self.store, self.emb, 3, 0.99, 2,
                            generator=lambda q, e: calls.append(q) or ("x [1]", {}))
        self.assertEqual((result["status"], calls), ("insufficient_evidence", []))

    def test_check_grounding_details(self):
        evidence = {1: "Leverage is debt divided by equity, 0.4871 in 2014.", 2: "Banks hold cash buffers."}
        self.assertEqual(ans.check_grounding("Leverage is debt divided by equity [1]. Banks hold cash buffers [2].", evidence, 0.6), [])
        self.assertEqual(ans.check_grounding("Leverage was 0.4871 in 2014. [1]", evidence, 0.6), [])  # trailing citation attaches
        self.assertIn("uncited", ans.check_grounding("Leverage is debt [1]. Banks hold cash.", evidence, 0.6)[0])
        self.assertIn("numbers", ans.check_grounding("Leverage was 0.52 in 2014 [1].", evidence, 0.6)[0])
        self.assertIn("content words", ans.check_grounding("Banks hold cash buffers [1].", evidence, 0.6)[0])
        self.assertEqual(ans.check_grounding("", evidence, 0.6), ["empty answer"])

    def test_ollama_generator_and_status_use_injected_transport(self):
        calls = []
        reply = {"response": " ok [1] ", "prompt_eval_count": 12, "eval_count": 3, "total_duration": 2_500_000_000}
        gen = ans.ollama_generator("m", options={"num_ctx": 4096}, post=lambda url, payload: calls.append((url, payload)) or reply)
        text, info = gen("q", [{"rank": 1, "source": "s", "section": "x", "text": "t"}])
        self.assertEqual((text, info["seconds"], info["prompt_tokens"]), ("ok [1]", 2.5, 12))
        url, payload = calls[0]
        self.assertTrue(url.endswith("/api/generate"))
        self.assertIn("[1] (s - x)", payload["prompt"])
        self.assertEqual((payload["options"]["temperature"], payload["options"]["seed"], payload["options"]["num_ctx"]), (0, 42, 4096))
        self.assertFalse(payload["stream"])
        responses = {"http://h/api/version": {"version": "1.0"}, "http://h/api/tags": {"models": [{"name": "m"}]}}
        self.assertTrue(ans.ollama_status("http://h", "m", get=responses.__getitem__)["available"])
        self.assertIn("not pulled", ans.ollama_status("http://h", "other", get=responses.__getitem__)["reason"])
        down = ans.ollama_status("http://h", "m", get=lambda u: (_ for _ in ()).throw(OSError("refused")))
        self.assertFalse(down["available"])


class TestRetrievalImprovements(unittest.TestCase):
    """D16: hybrid retrieval, section-aware embeddings, abstention gate, non-answers, question words."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        write_corpus(root)
        paths = docs.discover_sources(root, SETTINGS["sources"])
        self.chunks = docs.split_documents(docs.load_documents(paths, root), 200, 20)
        self.emb = BagOfWordsEmbedding()

    def tearDown(self):
        self.tmp.cleanup()

    def test_section_heading_is_embedded_when_enabled(self):
        doc = Document(page_content="Body text.", metadata={"section": "Risk Notes > Leverage"})
        self.assertEqual(idx.embedding_text(doc, True), "Risk Notes > Leverage\nBody text.")
        self.assertEqual(idx.embedding_text(doc, False), "Body text.")
        no_heading = Document(page_content="Body.", metadata={"section": "(no heading)"})
        self.assertEqual(idx.embedding_text(no_heading, True), "Body.")
        plain = idx.FaissStore.from_documents(self.chunks, self.emb)
        with_heading = idx.FaissStore.from_documents(self.chunks, self.emb, include_section=True)
        i = next(j for j, c in enumerate(self.chunks) if c.metadata["section"] != "(no heading)")
        self.assertFalse(np.allclose(plain.index.reconstruct(i), with_heading.index.reconstruct(i)))
        k = next(j for j, c in enumerate(self.chunks) if c.metadata["section"] == "(no heading)")
        self.assertTrue(np.allclose(plain.index.reconstruct(k), with_heading.index.reconstruct(k)))

    def test_modes_rank_and_keep_dense_scores(self):
        from fincrisis.rag.retrieval import Retriever
        store = idx.FaissStore.from_documents(self.chunks, self.emb, include_section=True)
        dense_cos = store.cosine_scores("leverage debt equity")
        self.assertEqual(len(dense_cos), len(self.chunks))
        for mode in ("dense", "lexical", "hybrid"):
            hits = Retriever(store, mode).search("leverage debt equity", 3)
            self.assertEqual([h["rank"] for h in hits], [1, 2, 3])
            self.assertEqual(hits[0]["section"], "Risk Notes > Leverage", mode)
            for h in hits:  # every hit carries its dense cosine, whatever the ranking
                i = next(j for j, d in enumerate(store.documents) if d.metadata["chunk_id"] == h["chunk_id"])
                self.assertAlmostEqual(h["score"], float(dense_cos[i]), places=6)
        r = Retriever(store, "hybrid")
        self.assertAlmostEqual(r.max_dense_score("leverage debt equity"), float(dense_cos.max()), places=6)
        with self.assertRaises(ValueError):
            Retriever(store, "other")

    def test_hybrid_fusion_promotes_lexical_only_match(self):
        from fincrisis.rag.retrieval import Retriever

        class Constant(BagOfWordsEmbedding):  # dense scores carry no information
            def _vector(self, text):
                return [1.0] + [0.0] * (self.size - 1)
        store = idx.FaissStore.from_documents(self.chunks, Constant())
        target = next(c.metadata["chunk_id"] for c in self.chunks if "Volatility clusters" in c.page_content)
        self.assertEqual(Retriever(store, "hybrid").search("volatility clusters persist", 1)[0]["chunk_id"], target)

    def test_question_level_abstention_gate_with_retriever(self):
        from fincrisis.rag.retrieval import Retriever
        store = idx.FaissStore.from_documents(self.chunks, self.emb)
        retriever = Retriever(store, "hybrid")
        answered = ans.answer("leverage debt equity", store, self.emb, 3, 0.1, 2, retriever=retriever)
        self.assertEqual(answered["status"], "answered")
        self.assertEqual(len(answered["evidence"]), 3)
        self.assertAlmostEqual(answered["top_score"], retriever.max_dense_score("leverage debt equity"))
        refused = ans.answer("leverage debt equity", store, self.emb, 3, 0.999, 2, retriever=retriever,
                             generator=lambda q, e: self.fail("model must not be called below the threshold"))
        self.assertEqual(refused["status"], "insufficient_evidence")

    def test_hedged_non_answers_count_as_abstention(self):
        for text in ("The test score is not directly stated in the evidence [1].",
                     "The documents do not mention the revenue [2].",
                     "No information is given about that."):
            self.assertTrue(ans.is_abstention(text), text)
        for text in ("The meanings of labels 0 and 1 are not documented by UCI [1].",
                     "Values above 1 are not assumed to be errors [2].",
                     "The VIX feature is never filled from a later observation [1]."):
            self.assertFalse(ans.is_abstention(text), text)

    def test_words_restated_from_the_question_are_not_claims(self):
        evidence = {1: "The first test session is 2015-01-02."}
        answer_text = "The market-stress test period begins on 2015-01-02 [1]."
        question = "On which date does the market-stress test period begin?"
        self.assertTrue(ans.check_grounding(answer_text, evidence, 0.6))           # without the question: rejected
        self.assertEqual(ans.check_grounding(answer_text, evidence, 0.6, question + " begins"), [])
        self.assertIn("numbers", ans.check_grounding("It begins on 2016-01-04 [1].", evidence, 0.6, question)[0])
        # word forms of question words do not count as claims; tiny remainders skip the ratio test
        self.assertEqual(ans.check_grounding("The label looks 20 days ahead [1].", {1: "H = 20 trading days"}, 0.6,
                                             "How many days ahead does the label look?"), [])
        unsupported = "The label measures volatility clustering regimes [1]."
        self.assertTrue(ans.check_grounding(unsupported, {1: "H = 20 trading days"}, 0.6, "What does the label measure?"))

    def test_default_dense_mode_uses_the_original_answer_path(self):
        store = idx.FaissStore.from_documents(self.chunks, self.emb)
        dense = {"retrieval": {"mode": "dense", "rrf_k": 60}}
        self.assertIsNone(pipeline.make_retriever(store, dense))
        self.assertEqual(pipeline.make_retriever(store, dense, "dense").mode, "dense")  # explicit mode: evaluation
        self.assertEqual(pipeline.make_retriever(store, {"retrieval": {"mode": "hybrid", "rrf_k": 60}}).mode, "hybrid")
        project = pipeline.load_rag_settings()["retrieval"]
        self.assertEqual((project["mode"], project["embed_section_headings"]), ("dense", False))

    def test_open_store_rejects_heading_setting_mismatch(self):
        root = Path(self.tmp.name)
        settings = {**SETTINGS, "index_dir": "index", "retrieval": {"embed_section_headings": True, "mode": "hybrid", "rrf_k": 60}}
        store, _, _ = build(root)  # manifest has no embed_section_headings -> treated as False
        with unittest.mock.patch.object(pipeline, "source_list", return_value=list(store_sources(root))):
            with self.assertRaises(idx.RAGIndexError):
                pipeline.open_store(settings, root, embeddings=BagOfWordsEmbedding())


def store_sources(root: Path):
    return [p.relative_to(root.resolve()).as_posix() for p in docs.discover_sources(root, SETTINGS["sources"])]


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
