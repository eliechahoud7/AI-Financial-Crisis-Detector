"""Tests for command-line scripts that had none (docs/decisions.md, D27).

Each script runs in a temporary project folder: `build_rag_index.py` and
`ask_rag.py` on a synthetic corpus with the fake embedding from test_rag;
`download_taiwan_bankruptcy.py` with its download replaced by an in-memory zip
(no network); `inspect_taiwan_bankruptcy.py` on a copy of the raw dataset
(skipped if absent). Nothing in the project is written.
"""

import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

from fincrisis.rag import answer as ans
from fincrisis.rag import index as idx
from fincrisis.rag import pipeline
from tests.test_rag import BagOfWordsEmbedding, build, write_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_main(module, *argv) -> tuple[int, str]:
    out = io.StringIO()
    with unittest.mock.patch.object(sys, "argv", ["script", *argv]), redirect_stdout(out):
        code = module.main()
    return code, out.getvalue()


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_mtime_ns for p in sorted(Path(root).rglob("*")) if p.is_file()}


RAG_SETTINGS = {"sources": ["docs/*.md", "docs/*.txt"], "exclude": [], "index_dir": "index", "embedding_model": "fake-bow",
                "chunk_size": 200, "chunk_overlap": 20, "retrieval": {"mode": "dense", "rrf_k": 60, "embed_section_headings": False},
                "top_k": 3, "min_relevance": 0.1, "max_answer_sentences": 2,
                "ollama": {"model": "m", "url": "http://localhost:1", "timeout_seconds": 1, "options": {}, "min_support": 0.6}}


class RagScriptCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_corpus(self.root)
        self.patches = [unittest.mock.patch.object(pipeline, "load_rag_settings", return_value=RAG_SETTINGS),
                        unittest.mock.patch.object(pipeline, "PROJECT_ROOT", self.root),
                        unittest.mock.patch.object(pipeline, "model_revision", return_value=None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()


class TestBuildRagIndex(RagScriptCase):
    def setUp(self):
        super().setUp()
        self.module = load_script("build_rag_index")
        self.embeddings = unittest.mock.patch.object(pipeline, "make_embeddings", return_value=BagOfWordsEmbedding())
        self.embeddings.start()
        self.addCleanup(self.embeddings.stop)

    def test_build_refuse_overwrite_then_force(self):
        code, out = run_main(self.module)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.root / "index" / "manifest.json").exists())
        manifest = json.loads((self.root / "index" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["sources"]), {"docs/risk.md", "docs/notes.txt"})
        before = tree(self.root / "index")
        code, out = run_main(self.module)
        self.assertEqual(code, 1)
        self.assertIn("Use --force", out)
        self.assertEqual(tree(self.root / "index"), before)
        self.assertEqual(run_main(self.module, "--force")[0], 0)

    def test_missing_embedding_model_writes_nothing(self):
        with unittest.mock.patch.object(pipeline, "make_embeddings",
                                        side_effect=RuntimeError("Embedding model fake-bow is not in the local cache")):
            code, out = run_main(self.module)
        self.assertEqual(code, 1)
        self.assertIn("not in the local cache", out)
        self.assertIn("Nothing was written", out)
        self.assertFalse((self.root / "index").exists())

    def test_no_documents_writes_nothing(self):
        shutil.rmtree(self.root / "docs")
        code, out = run_main(self.module)
        self.assertEqual(code, 1)
        self.assertIn("no documents match", out)
        self.assertFalse((self.root / "index").exists())


class TestAskRag(RagScriptCase):
    def setUp(self):
        super().setUp()
        store = build(self.root)[0]
        self.open = unittest.mock.patch.object(pipeline, "open_store", return_value=(store, {}, store.embeddings))
        self.open.start()
        self.addCleanup(self.open.stop)
        self.module = load_script("ask_rag")

    def test_answer_with_sources_and_json(self):
        code, out = run_main(self.module, "What is liquidity risk?")
        self.assertEqual(code, 0)
        self.assertIn("Liquidity risk is the danger", out)
        self.assertIn("Sources:\n  [1] docs/risk.md", out)
        self.assertIn("Method: extractive", out)
        code, out = run_main(self.module, "What is liquidity risk?", "--json")
        result = json.loads(out)
        self.assertEqual((result["status"], result["citations"][0]["source"]), ("answered", "docs/risk.md"))

    def test_abstention_shows_the_best_score_and_threshold(self):
        code, out = run_main(self.module, "zzz qqq xxx?")
        self.assertEqual(code, 0)
        self.assertIn(ans.INSUFFICIENT, out)
        self.assertIn("threshold 0.1", out)

    def test_unavailable_local_model_is_reported(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "no server"}):
            code, out = run_main(self.module, "What is liquidity risk?", "--llm", "ollama")
        self.assertEqual(code, 0)
        self.assertIn("Warning: local model unavailable (no server); extractive answer returned", out)

    def test_missing_or_stale_index_fails_cleanly(self):
        with unittest.mock.patch.object(pipeline, "open_store",
                                        side_effect=idx.RAGIndexError("index is out of date for ['docs/risk.md']")):
            code, out = run_main(self.module, "What is liquidity risk?")
        self.assertEqual(code, 1)
        self.assertIn("Failed: index is out of date", out)


def csv_bytes(rows: int, columns: int) -> bytes:
    header = ["Bankrupt?"] + [f" f{i}" for i in range(1, columns)]
    body = [[str(r % 2)] + [str(r + c) for c in range(1, columns)] for r in range(rows)]
    return "\n".join(",".join(row) for row in [header, *body]).encode("utf-8") + b"\n"


def zipped(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class TestDownloadTaiwan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        shutil.copy2(PROJECT_ROOT / "config" / "settings.yaml", self.root / "config" / "settings.yaml")
        self.module = load_script("download_taiwan_bankruptcy")
        self.data = csv_bytes(4, 3)
        for name, value in (("PROJECT_ROOT", self.root), ("EXPECTED_ROWS", 4), ("EXPECTED_COLUMNS", 3)):
            patcher = unittest.mock.patch.object(self.module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.out = self.root / "data" / "raw" / "taiwan_bankruptcy"

    def tearDown(self):
        self.tmp.cleanup()

    def run_with(self, archive: bytes, *argv):
        with unittest.mock.patch.object(self.module, "download", return_value=archive) as download:
            code, out = run_main(self.module, *argv)
        return code, out, download

    def test_download_validates_and_records_checksums(self):
        code, out, download = self.run_with(zipped({"data.csv": self.data, "../evil.txt": b"x"}))
        self.assertEqual(code, 0, out)
        download.assert_called_once_with(self.module.URL)
        self.assertEqual((self.out / "data.csv").read_bytes(), self.data)
        meta = json.loads((self.out / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["csv_sha256"], self.module.sha256(self.data))
        self.assertEqual(meta["target_value_counts"], {"0": 2, "1": 2})
        self.assertEqual(sorted(p.name for p in self.root.rglob("*") if p.is_file()),
                         ["data.csv", "metadata.json", "settings.yaml"])  # only the named member is read

    def test_existing_download_is_never_touched_without_force(self):
        self.run_with(zipped({"data.csv": self.data}))
        before = tree(self.out)
        code, out, download = self.run_with(b"unused")
        self.assertEqual(code, 0)
        download.assert_not_called()
        (self.out / "metadata.json").write_text("{}", encoding="utf-8")  # metadata lost
        code, out, _ = self.run_with(b"unused")
        self.assertEqual(code, 1)
        self.assertIn("missing or invalid", out)
        (self.out / "metadata.json").write_text(json.dumps({"csv_sha256": "0" * 64}), encoding="utf-8")
        code, out, _ = self.run_with(b"unused")
        self.assertEqual(code, 1)
        self.assertIn("does not match", out)
        self.assertEqual((self.out / "data.csv").read_bytes(), self.data)
        self.assertIn("data.csv", before)

    def test_invalid_downloads_write_nothing(self):
        for archive, message in ((zipped({"data.csv": csv_bytes(5, 3)}), "Expected 4 rows"),
                                 (zipped({"other.csv": self.data}), "data.csv not in archive")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.run_with(archive)
            self.assertFalse(self.out.exists())


RAW = PROJECT_ROOT / "data" / "raw" / "taiwan_bankruptcy"


@unittest.skipUnless((RAW / "data.csv").exists(), "raw Taiwan dataset not present")
class TestInspectTaiwan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "config").mkdir()
        shutil.copy2(PROJECT_ROOT / "config" / "settings.yaml", self.root / "config" / "settings.yaml")
        shutil.copytree(RAW, self.root / "data" / "raw" / "taiwan_bankruptcy")
        self.module = load_script("inspect_taiwan_bankruptcy")
        patcher = unittest.mock.patch.object(self.module, "PROJECT_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.reports = self.root / "data" / "reports" / "taiwan_bankruptcy"
        self.raw = self.root / "data" / "raw" / "taiwan_bankruptcy" / "data.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_report_written_raw_unchanged_and_no_overwrite(self):
        before = self.raw.read_bytes()
        code, out = run_main(self.module)
        self.assertEqual(code, 0, out)
        self.assertTrue((self.reports / "inspection_report.md").exists())
        self.assertTrue((self.reports / "feature_summary.csv").exists())
        self.assertEqual(self.raw.read_bytes(), before)
        code, out = run_main(self.module)
        self.assertEqual(code, 1)
        self.assertIn("Use --force", out)

    def test_altered_dataset_is_refused(self):
        self.raw.write_bytes(self.raw.read_bytes() + b"\n")
        code, out = run_main(self.module)
        self.assertEqual(code, 1)
        self.assertIn("Checksum mismatch", out)
        self.assertFalse(self.reports.exists())


if __name__ == "__main__":
    unittest.main()
