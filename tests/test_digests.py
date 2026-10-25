"""Tests for embedding and reranking long hex digests as a placeholder (docs/decisions.md, D33).

Unit tests use fake embeddings and rerankers. The project test uses the real
index and models and is skipped if they are not available.
"""

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from langchain_core.documents import Document

from fincrisis.dashboard import loaders as ld
from fincrisis.rag import answer as ans
from fincrisis.rag import documents as docs
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline
from fincrisis.rag.retrieval import Retriever, rerank_text
from tests.test_rag import MODEL, PROJECT_ROOT, BagOfWordsEmbedding, model_cached, write_corpus

SHA256 = "67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8"
ROW = f"| CSV SHA-256 | {SHA256} |"


class Recording(BagOfWordsEmbedding):
    """Records the texts it embeds."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return super().embed_documents(texts)


class TestMask(unittest.TestCase):
    def test_long_hex_digests_are_masked_and_short_ids_kept(self):
        md5, git = "d41d8cd98f00b204e9800998ecf8427e", "1a96172d" * 5
        text = f"sha {SHA256}, md5 {md5}, full commit {git}, short commit 1a96172, 31 hex {SHA256[:31]}"
        self.assertEqual(idx.mask_digests(text),
                         f"sha <sha256>, md5 <sha256>, full commit <sha256>, short commit 1a96172, 31 hex {SHA256[:31]}")
        self.assertEqual(idx.mask_digests(SHA256.upper()), "<sha256>")
        self.assertEqual(idx.mask_digests("no digest here"), "no digest here")

    def test_the_index_embeds_masked_text_but_stores_the_real_text(self):
        document = Document(page_content=ROW, metadata={"source": "s.md", "section": "Input", "chunk_id": "s.md#1"})
        embedding = Recording()
        store = idx.FaissStore.from_documents([document], embedding, include_section=True, mask=True)
        self.assertEqual(embedding.seen, ["Input\n| CSV SHA-256 | <sha256> |"])
        self.assertEqual(store.documents[0].page_content, ROW)  # evidence and citations keep the digest
        plain = Recording()
        idx.FaissStore.from_documents([document], plain)
        self.assertEqual(plain.seen, [ROW])  # off unless configured

    def test_the_reranker_and_the_sentence_scorer_read_masked_text(self):
        document = Document(page_content=ROW, metadata={"source": "s.md", "section": "Input", "chunk_id": "s.md#1"})
        self.assertEqual(rerank_text(document, mask=True), "s.md - Input\n| CSV SHA-256 | <sha256> |")
        self.assertEqual(rerank_text(document), f"s.md - Input\n{ROW}")
        read = []
        def reranker(question, passages):
            read.extend(passages)
            return [1.0] * len(passages)
        store = idx.FaissStore.from_documents([document], BagOfWordsEmbedding())
        hits = Retriever(store, "rerank", reranker=reranker, candidates=1, mask=True).search("checksum", 1)
        self.assertEqual(read, ["s.md - Input\n| CSV SHA-256 | <sha256> |"])
        self.assertIn(SHA256, hits[0]["text"])
        evidence = [{"rank": 1, "source": "s.md", "section": "Input", "chunk_id": "s.md#1", "score": 0.5,
                     "text": f"The raw CSV has the checksum {SHA256} recorded in the metadata."}]
        scored = []
        text, _ = ans.extractive_answer("checksum?", evidence, BagOfWordsEmbedding(), 1,
                                        lambda q, ps: scored.extend(ps) or [1.0] * len(ps), mask=True)
        self.assertEqual(scored, ["s.md - Input\nThe raw CSV has the checksum <sha256> recorded in the metadata."])
        self.assertIn(SHA256, text)  # the answer quotes the sentence as written

    def test_make_retriever_passes_the_setting(self):
        settings = {"retrieval": {"mode": "rerank", "rrf_k": 60, "rerank_model": "m", "rerank_candidates": 5,
                                  "mask_digests": True}}
        with unittest.mock.patch.object(pipeline, "make_reranker", return_value=lambda q, ps: [0.0] * len(ps)):
            store = idx.FaissStore.from_documents([Document(page_content="Checksums are recorded.", metadata={})],
                                                  BagOfWordsEmbedding())
            self.assertTrue(pipeline.make_retriever(store, settings).mask)
            settings["retrieval"]["mask_digests"] = False
            self.assertFalse(pipeline.make_retriever(store, settings).mask)
        self.assertTrue(pipeline.load_rag_settings()["retrieval"]["mask_digests"])


class TestManifest(unittest.TestCase):
    SETTINGS = {"sources": ["docs/*.md", "docs/*.txt"], "index_dir": "index", "embedding_model": "fake-bow",
                "chunk_size": 200, "chunk_overlap": 20,
                "retrieval": {"embed_section_headings": False, "mask_digests": True}}

    def build(self, root: Path, mask: bool | None):
        loaded = docs.load_documents(docs.discover_sources(root, self.SETTINGS["sources"]), root)
        chunks = docs.split_documents(loaded, 200, 20)
        store = idx.FaissStore.from_documents(chunks, BagOfWordsEmbedding(), mask=bool(mask))
        manifest = {"embedding_model": "fake-bow", "chunks": len(chunks),
                    "sources": {d.metadata["source"]: d.metadata["sha256"] for d in loaded}}
        if mask is not None:
            manifest["mask_digests"] = mask
        idx.save_store(store, root / "index", manifest, force=True)

    def test_an_index_built_with_another_setting_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            self.build(root, True)
            self.assertTrue(ld.rag_index_status(self.SETTINGS, root)["ready"])
            pipeline.open_store(self.SETTINGS, root, BagOfWordsEmbedding())
            for built in (False, None):  # an index built before D33 has no setting: not masked
                self.build(root, built)
                self.assertIn("different mask_digests", ld.rag_index_status(self.SETTINGS, root)["problem"])
                with self.assertRaisesRegex(idx.RAGIndexError, "mask_digests"):
                    pipeline.open_store(self.SETTINGS, root, BagOfWordsEmbedding())


INDEX = PROJECT_ROOT / pipeline.load_rag_settings()["index_dir"]
RERANKER = pipeline.load_rag_settings()["retrieval"]["rerank_model"]


@unittest.skipUnless(INDEX.exists() and model_cached(MODEL) and model_cached(RERANKER), "index or models not available")
class TestProjectIndex(unittest.TestCase):
    def test_the_model_files_passage_reaches_the_reranker(self):
        """Regression (D33): 55% of this passage was digest tokens, and it ranked 30th, outside the candidates."""
        rag = pipeline.load_rag_settings()
        if not ld.rag_index_status(rag)["ready"]:
            self.skipTest("RAG index not current")
        store, manifest, _ = pipeline.open_store(rag)
        self.assertTrue(manifest["mask_digests"])
        passage = next(d for d in store.documents if "artifacts/models/taiwan_bankruptcy/`:" in d.page_content)
        self.assertRegex(passage.page_content, r"[0-9a-f]{64}")  # stored with its real digests
        hits = pipeline.make_retriever(store, rag).search("Where are the trained company-distress model files saved?",
                                                          rag["retrieval"]["rerank_candidates"])
        self.assertIn(passage.metadata["chunk_id"], [h["chunk_id"] for h in hits])


if __name__ == "__main__":
    unittest.main()
