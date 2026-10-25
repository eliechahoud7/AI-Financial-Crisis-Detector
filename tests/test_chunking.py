"""Tests for structure-aware, token-bounded chunking (docs/decisions.md, D30).

Unit tests count tokens as whitespace-separated words plus two special tokens,
so they need no model. The tests on the project documents use the real
embedding model's tokenizer and are skipped if it is not in the local cache.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from fincrisis.dashboard import loaders as ld
from fincrisis.rag import chunking as chk
from fincrisis.rag import documents as docs
from fincrisis.rag import evaluation as ev
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline
from tests.test_rag import MODEL, PROJECT_ROOT, BagOfWordsEmbedding, model_cached, write_corpus


def count(text: str) -> int:
    """Whitespace words plus [CLS] and [SEP]."""
    return len(text.split()) + 2


def doc(text: str, source: str = "docs/test.md") -> Document:
    return Document(page_content=text, metadata={"source": source, "sha256": "x"})


def split(text: str, max_tokens: int = 40, min_tokens: int = 8, source: str = "docs/test.md", **kwargs):
    return chk.split_structured([doc(text, source)], count, max_tokens, min_tokens, **kwargs)


def sentences(n: int, start: int = 0) -> str:
    return " ".join(f"Sentence number {i} says something useful." for i in range(start, start + n))


def fence_count(text: str) -> int:
    return sum(line.strip().startswith("```") for line in text.splitlines())


class TestProse(unittest.TestCase):
    def test_short_section_is_one_chunk_with_its_text_and_lines(self):
        text = "# Title\n\n## Part\n\nFirst paragraph\nwraps here.\n\nSecond paragraph.\n"
        [chunk] = split(text)
        self.assertEqual(chunk.page_content, "# Title\n\n## Part\n\nFirst paragraph\nwraps here.\n\nSecond paragraph.")
        self.assertEqual(chunk.metadata["section"], "Title > Part")  # the bare title introduces its subsection
        self.assertEqual((chunk.metadata["start_line"], chunk.metadata["end_line"]), (1, 8))
        self.assertEqual(chunk.metadata["chunk_id"], "docs/test.md#1")

    def test_long_paragraph_splits_between_sentences_only(self):
        text = "## Notes\n\n" + sentences(30) + "\n"
        chunks = split(text)
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertLessEqual(count(chunk.page_content), 40)
            body = chunk.page_content.split("\n")[-1]
            self.assertTrue(body.startswith("Sentence number") and body.endswith("useful."), body)
        found = re.findall(r"Sentence number (\d+)", " ".join(c.page_content for c in chunks))
        self.assertEqual(found, [str(i) for i in range(30)])  # every sentence once, in order

    def test_over_long_sentence_falls_back_to_words(self):
        words = [f"w{i}" for i in range(100)]
        chunks = split("## S\n\n" + " ".join(words) + ".\n")
        self.assertTrue(all(count(c.page_content) <= 40 for c in chunks))
        self.assertEqual(" ".join(c.page_content for c in chunks).replace("## S", "").split(), words[:-1] + ["w99."])

    def test_plain_text_files_have_no_markdown_structure(self):
        chunks = split("# not a heading\n\n| not | a table |\n", source="docs/notes.txt")
        self.assertEqual([c.metadata["section"] for c in chunks], ["(no heading)"])


class TestTokenLimit(unittest.TestCase):
    MIXED = ("# Doc\n\n## A\n\n" + sentences(20) + "\n\n| h1 | h2 |\n|---|---|\n"
             + "".join(f"| row {i} | value {i} |\n" for i in range(40))
             + "\n```python\n" + "".join(f"x{i} = {i}\n" for i in range(60)) + "```\n\n- item " + sentences(12) + "\n")

    def test_every_chunk_fits_including_the_section_prefix(self):
        for include in (False, True):
            chunks = split(self.MIXED, include_section=include)
            for chunk in chunks:
                prefix = f"{chunk.metadata['section']}\n" if include else ""
                self.assertLessEqual(count(prefix + chunk.page_content), 40, chunk.page_content)

    def test_invalid_limits_are_refused(self):
        for max_tokens, min_tokens in ((0, 0), (10, 10), (10, -1)):
            with self.assertRaises(ValueError):
                chk.split_structured([], count, max_tokens, min_tokens)


class TestTables(unittest.TestCase):
    TABLE = "## Results\n\n| Model | Score |\n|---|---|\n" + "".join(f"| m{i} | {i}.5 |\n" for i in range(30))

    def test_split_table_repeats_its_header_and_keeps_rows_whole(self):
        chunks = split(self.TABLE)
        self.assertGreater(len(chunks), 2)
        rows = []
        for chunk in chunks:
            lines = [l for l in chunk.page_content.split("\n") if l.startswith("|")]
            self.assertEqual(lines[:2], ["| Model | Score |", "|---|---|"])
            self.assertTrue(chunk.page_content.startswith("## Results\n\n| Model | Score |"))  # says what the rows are
            rows += lines[2:]
        self.assertEqual(rows, [f"| m{i} | {i}.5 |" for i in range(30)])
        sizes = [len([l for l in c.page_content.split("\n") if l.startswith("| m")]) for c in chunks]
        self.assertLessEqual(max(sizes) - min(sizes), 2)  # rows spread evenly: no short remainder

    def test_heading_stays_with_the_first_rows(self):
        first = split(self.TABLE)[0].page_content
        self.assertTrue(first.startswith("## Results\n\n| Model | Score |"), first)
        self.assertIn("| m0 | 0.5 |", first)

    def test_short_introduction_stays_with_the_table(self):
        text = "## Features\n\nEach feature uses past closes only.\n\n" + self.TABLE.split("\n\n", 1)[1]
        first = split(text, min_tokens=16)[0].page_content  # the introduction alone would be a short chunk
        self.assertIn("Each feature uses past closes only.", first)
        self.assertIn("| m0 | 0.5 |", first)


class TestCodeAndDefinitions(unittest.TestCase):
    def test_definition_lead_in_formula_and_continuation_stay_together(self):
        filler = "## D2. Target\n\n" + sentences(4) + "\n\n"
        definition = ("- **Decision:** For each day *t*:\n\n  ```\n  y_t = 1 if DD_t <= -0.05\n"
                      "        0 otherwise\n  ```\n\n  with `H = 20` trading days.\n")
        chunks = split(filler + definition, max_tokens=60)
        [chunk] = [c for c in chunks if "y_t = 1" in c.page_content]
        for part in ("For each day *t*:", "        0 otherwise", "with `H = 20` trading days."):
            self.assertIn(part, chunk.page_content)  # indentation inside code is kept
        self.assertEqual(fence_count(chunk.page_content), 2)

    def test_lead_in_with_the_colon_inside_emphasis_stays_with_its_list(self):
        """Regression (D30): "- **Safeguards (added later):**" ends in ":**", and was cut from the rule it introduces."""
        filler = "## D13. Models\n\n" + "\n".join(f"- Item {i} says something useful here." for i in range(5)) + "\n"
        text = filler + ("- **Safeguards (added later):**\n  - **Primary model:** the model with the highest score.\n"
                         "  - **Locked:** the selection cannot be replaced.\n")
        chunks = split(text, max_tokens=50)
        [chunk] = [c for c in chunks if "Primary model:" in c.page_content]
        self.assertIn("- **Safeguards (added later):**", chunk.page_content)
        for marker in ("Label:**", "Label:*", "Label:__", "Label:"):
            self.assertTrue(chk._blocks([(1, f"- **{marker}"), (2, "  - child")], True)[0].glue_next, marker)
        inline = chk._blocks([(1, "- **Primary metric:** average precision.")], True)[0]
        self.assertFalse(inline.glue_next)  # text after the colon: not a lead-in

    def test_long_code_block_is_re_fenced_in_every_part(self):
        text = "## Code\n\n```python\n" + "".join(f"value_{i} = {i}\n" for i in range(80)) + "```\n"
        chunks = split(text)
        self.assertGreater(len(chunks), 2)
        for chunk in chunks:
            lines = chunk.page_content.split("\n")
            self.assertEqual(fence_count(chunk.page_content), 2)
            self.assertEqual(lines[-1], "```")
            self.assertIn("```python", lines)
        values = re.findall(r"value_(\d+)", " ".join(c.page_content for c in chunks))
        self.assertEqual(values, [str(i) for i in range(80)])

    def test_headings_inside_code_blocks_are_not_sections(self):
        [chunk] = split("## Real\n\n```\n# comment, not a heading\n```\n", max_tokens=60)
        self.assertEqual(chunk.metadata["section"], "Real")


class TestFragments(unittest.TestCase):
    def test_short_trailing_fragment_is_merged(self):
        text = "## S\n\n" + "\n\n".join(sentences(1, i) for i in range(9)) + "\n\nTail.\n"
        chunks = split(text, max_tokens=40, min_tokens=10)
        self.assertTrue(all(count(c.page_content) >= 10 for c in chunks), [c.page_content for c in chunks])
        self.assertIn("Tail.", chunks[-1].page_content)

    def test_no_chunk_is_only_a_heading(self):
        text = ("# Report\n\n## Data\n\n| a | b |\n|---|---|\n" + "".join(f"| {i} | {i} |\n" for i in range(40))
                + "\n### Notes\n\n" + sentences(3) + "\n")
        for chunk in split(text):
            content = [l for l in chunk.page_content.split("\n") if l.strip() and not l.startswith("#")]
            self.assertTrue(content, chunk.page_content)

    def test_complete_short_sections_are_kept(self):
        chunks = split("## A\n\nNone.\n\n## B\n\n" + sentences(3) + "\n")
        self.assertEqual([c.metadata["section"] for c in chunks], ["A", "B"])  # siblings are never merged


class TestCitations(unittest.TestCase):
    def test_ids_sections_and_line_ranges_point_at_the_source(self):
        text = ("# Guide\n\nIntro sentence for the guide.\n\n## Setup\n\n" + sentences(12) + "\n\n## Use\n\n"
                + "| k | v |\n|---|---|\n" + "".join(f"| key{i} | val{i} |\n" for i in range(25)))
        source_lines = text.split("\n")
        chunks = split(text) + split(text, source="docs/other.md")
        for name in ("docs/test.md", "docs/other.md"):
            ids = [c.metadata["chunk_id"] for c in chunks if c.metadata["source"] == name]
            self.assertEqual(ids, [f"{name}#{i}" for i in range(1, len(ids) + 1)])
        for chunk in chunks:
            start, end = chunk.metadata["start_line"], chunk.metadata["end_line"]
            span = source_lines[start - 1:end]
            header = {"## Use", "| k | v |", "|---|---|"}  # repeated in every part of a split table
            for line in chunk.page_content.split("\n"):
                if line.strip() and line not in header:  # a paragraph split between sentences gives part of a line
                    self.assertTrue(any(line in source for source in span), (chunk.metadata, line))
            heading = [l for l in source_lines[:start] if l.startswith("## ")]
            if heading:
                self.assertTrue(chunk.metadata["section"].endswith(heading[-1][3:]), chunk.metadata)


class FakeTokenizer:
    def __call__(self, text, verbose=True):
        return {"input_ids": [0] * count(text)}


class FakeClient:
    tokenizer, max_seq_length = FakeTokenizer(), 30


class TokenizedBagOfWords(BagOfWordsEmbedding):
    _client = FakeClient()


class TestPipeline(unittest.TestCase):
    def test_method_defaults_and_manifest_matching(self):
        self.assertEqual(pipeline.chunking({})["method"], "characters")  # settings written before D30
        project = pipeline.load_rag_settings()
        self.assertEqual(pipeline.chunking(project)["method"], "structured")
        self.assertTrue(pipeline.chunking_matches({}, {}))  # an index built before D30 used characters
        built = {"chunking": {"method": "structured", "max_tokens": 256, "min_tokens": 64}}
        self.assertTrue(pipeline.chunking_matches(built, project))
        self.assertFalse(pipeline.chunking_matches({}, project))
        self.assertFalse(pipeline.chunking_matches({"chunking": {**built["chunking"], "min_tokens": 32}}, project))

    def test_split_uses_the_model_tokenizer_and_limit(self):
        settings = {"chunking": {"method": "structured", "min_tokens": 5}, "retrieval": {"embed_section_headings": False}}
        chunks, record = pipeline.split(settings, [doc("## S\n\n" + sentences(20))], TokenizedBagOfWords())
        self.assertEqual(record, {"method": "structured", "max_tokens": 30, "min_tokens": 5})
        self.assertTrue(all(count(c.page_content) <= 30 for c in chunks))
        with self.assertRaises(idx.RAGIndexError):  # an embedding without a tokenizer cannot be counted
            pipeline.split(settings, [doc("text")], BagOfWordsEmbedding())
        with self.assertRaises(ValueError):
            pipeline.split({"chunking": {"method": "other"}}, [], TokenizedBagOfWords())
        chunks, record = pipeline.split({"chunk_size": 200, "chunk_overlap": 20}, [doc("## S\n\nText.")], None)
        self.assertEqual(record, {"method": "characters"})

    def test_an_index_with_other_chunking_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            settings = {"sources": ["docs/*.md", "docs/*.txt"], "index_dir": "index", "embedding_model": "fake-bow",
                        "chunking": {"method": "structured", "min_tokens": 5}, "chunk_size": 200, "chunk_overlap": 20,
                        "retrieval": {"embed_section_headings": False}}
            embeddings = TokenizedBagOfWords()
            loaded = docs.load_documents(docs.discover_sources(root, settings["sources"]), root)
            chunks, record = pipeline.split(settings, loaded, embeddings)
            store = idx.FaissStore.from_documents(chunks, embeddings)
            idx.save_store(store, root / "index", {"embedding_model": "fake-bow", "chunks": len(chunks), "chunking": record,
                                                   "sources": {d.metadata["source"]: d.metadata["sha256"] for d in loaded}},
                           force=False)
            self.assertEqual(len(pipeline.open_store(settings, root, embeddings)[0].documents), len(chunks))
            characters = {k: v for k, v in settings.items() if k != "chunking"}
            with self.assertRaises(idx.RAGIndexError):
                pipeline.open_store(characters, root, embeddings)
            self.assertTrue(ld.rag_index_status(settings, root)["ready"])
            self.assertIn("different chunking", ld.rag_index_status(characters, root)["problem"])


@unittest.skipUnless(model_cached(MODEL), "embedding model not in the local cache")
class TestProjectDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = pipeline.load_rag_settings()
        cls.embeddings = pipeline.make_embeddings(MODEL)
        counter, cls.limit = pipeline.token_counter(cls.embeddings)
        cls.count = staticmethod(counter)
        paths = docs.discover_sources(PROJECT_ROOT, ["README.md", "docs/*.md"])  # committed documents only
        cls.chunks, _ = pipeline.split(cls.settings, docs.load_documents(paths, PROJECT_ROOT), cls.embeddings)

    def test_the_limit_is_the_model_input_length_with_special_tokens(self):
        self.assertEqual(self.limit, 256)
        self.assertEqual(self.count(""), 2)  # [CLS] and [SEP]

    def test_no_chunk_is_cut_off_by_the_embedding_model(self):
        self.assertTrue(all(self.count(c.page_content) <= self.limit for c in self.chunks))
        self.assertTrue(all(fence_count(c.page_content) % 2 == 0 for c in self.chunks))

    def test_the_market_stress_definition_is_one_chunk(self):
        [chunk] = [c for c in self.chunks if "DD_t = min over" in c.page_content]
        self.assertIn("y_t  = 1  if DD_t <= -0.05", chunk.page_content)
        self.assertIn("In words: `y_t = 1` if, during the next 20 trading days", chunk.page_content)
        self.assertEqual(chunk.metadata["section"], "Design Decisions > D2. Market-stress target definition")

    def test_no_chunk_ends_with_a_lead_in_whose_list_continues_in_the_next(self):
        for chunk, following in zip(self.chunks, self.chunks[1:]):
            last = chunk.page_content.rstrip().split("\n")[-1]
            same_section = chunk.metadata["source"] == following.metadata["source"] and \
                chunk.metadata["section"] == following.metadata["section"]
            lead_in = re.search(r":[*_]*$", last.rstrip())  # independent of chk.LEAD_IN, so a broken rule is caught
            self.assertFalse(same_section and lead_in, (chunk.metadata["chunk_id"], last))

    def test_development_questions_are_valid_for_the_new_chunks(self):
        chunk_dicts = [{"source": c.metadata["source"], "text": c.page_content} for c in self.chunks]
        questions = json.loads((PROJECT_ROOT / self.settings["eval_set"]).read_text(encoding="utf-8"))["questions"]
        self.assertEqual(ev.validate_questions(questions, chunk_dicts), [])


if __name__ == "__main__":
    unittest.main()
