"""Tests for the RAG evaluation scripts and the validation-set guard (docs/decisions.md, D16, D24).

The scripts run on a small synthetic corpus in a temporary folder, with the
fake embedding from test_rag and, for generation, a fake local model; no index,
model or Ollama server of the project is touched and no project report is
written. The validation set itself is only hashed or copied, never evaluated.
"""

import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stdout
from pathlib import Path

from fincrisis.qa import validation_guard as guard
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline
from tests.test_rag import build, write_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV = "eval/dev_questions.json"
QUESTIONS = {"description": "synthetic development set", "questions": [
    {"id": "q1", "question": "What is liquidity risk?", "relevant": [{"source": "docs/risk.md", "contains": "Liquidity risk"}]},
    {"id": "q2", "question": "What does leverage measure?", "relevant": [{"source": "docs/risk.md", "contains": "debt"}]},
    {"id": "u1", "question": "Who won the 2018 FIFA World Cup?", "answerable": False}]}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_validation_set(root: Path, relative: str = guard.VALIDATION_SET) -> None:
    (root / relative).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / guard.VALIDATION_SET, root / relative)


class TestGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / DEV).parent.mkdir(parents=True)
        (self.root / DEV).write_text(json.dumps(QUESTIONS), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_development_set_is_the_default(self):
        self.assertEqual(guard.resolve_question_set(None, DEV, self.root), (self.root / DEV, ""))

    def test_validation_set_needs_the_flag_and_its_checksum(self):
        copy_validation_set(self.root)
        with self.assertRaisesRegex(guard.QuestionSetError, "must not be used to tune"):
            guard.resolve_question_set(guard.VALIDATION_SET, DEV, self.root)
        path, suffix = guard.resolve_question_set(guard.VALIDATION_SET, DEV, self.root, final_validation=True)
        self.assertEqual((path, suffix), (self.root / guard.VALIDATION_SET, "_validation_questions"))
        questions = json.loads(path.read_text(encoding="utf-8"))
        questions["questions"][0]["question"] += " (edited)"
        path.write_text(json.dumps(questions), encoding="utf-8")
        with self.assertRaisesRegex(guard.QuestionSetError, "no longer matches its recorded checksum"):
            guard.resolve_question_set(guard.VALIDATION_SET, DEV, self.root, final_validation=True)

    def test_a_renamed_copy_is_recognised(self):
        copy_validation_set(self.root, "eval/my_questions.json")
        with self.assertRaisesRegex(guard.QuestionSetError, "is the RAG validation set"):
            guard.resolve_question_set("eval/my_questions.json", DEV, self.root)
        crlf = self.root / "eval" / "crlf_copy.json"  # line endings do not hide it either
        crlf.write_bytes((self.root / "eval/my_questions.json").read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(guard.QuestionSetError, "is the RAG validation set"):
            guard.resolve_question_set("eval/crlf_copy.json", DEV, self.root)

    def test_flag_with_another_set_and_missing_files_are_refused(self):
        with self.assertRaisesRegex(guard.QuestionSetError, "applies only to the validation set"):
            guard.resolve_question_set(None, DEV, self.root, final_validation=True)
        with self.assertRaisesRegex(guard.QuestionSetError, "not found"):
            guard.resolve_question_set("eval/none.json", DEV, self.root)

    def test_missing_manifest_refuses_rather_than_allows(self):
        with unittest.mock.patch.object(guard.loaders, "FROZEN_MANIFEST", self.root / "nowhere.json"):
            with self.assertRaisesRegex(guard.QuestionSetError, "checksum is unavailable"):
                guard.resolve_question_set(None, DEV, self.root)


class ScriptCase(unittest.TestCase):
    """Runs a script's main() against a synthetic corpus, index and settings in a temporary folder."""

    script = ""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        write_corpus(self.root)
        (self.root / DEV).parent.mkdir(parents=True)
        (self.root / DEV).write_text(json.dumps(QUESTIONS), encoding="utf-8")
        store, manifest, _ = build(self.root)
        manifest = {**manifest, "embedding_model_revision": None, "chunk_size": 200, "chunk_overlap": 20}
        self.settings = {
            "eval_set": DEV, "eval_k": [1, 3], "min_relevance": 0.3, "top_k": 3, "max_answer_sentences": 2,
            "reports_dir": "reports", "retrieval": {"mode": "dense", "rrf_k": 60, "embed_section_headings": False},
            "ollama": {"model": "fake", "url": "http://localhost:1", "timeout_seconds": 1, "min_support": 0.6,
                       "options": {}}}
        self.opened = []

        def open_store(settings, *args, **kwargs):
            self.opened.append(True)
            return store, manifest, store.embeddings
        self.patches = [unittest.mock.patch.object(pipeline, "load_rag_settings", return_value=self.settings),
                        unittest.mock.patch.object(pipeline, "PROJECT_ROOT", self.root),
                        unittest.mock.patch.object(pipeline, "open_store", open_store)]
        for p in self.patches:
            p.start()
        self.module = load_script(self.script)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_main(self, *argv) -> tuple[int, str]:
        out = io.StringIO()
        with unittest.mock.patch.object(sys, "argv", [self.script, *argv]), redirect_stdout(out):
            code = self.module.main()
        return code, out.getvalue()

    def reports(self) -> list[str]:
        directory = self.root / "reports"
        return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


class TestRetrievalEvaluationScript(ScriptCase):
    script = "evaluate_rag"

    def test_development_set_report(self):
        code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.reports(), ["retrieval_eval.json", "retrieval_eval.md"])
        result = json.loads((self.root / "reports" / "retrieval_eval.json").read_text(encoding="utf-8"))
        self.assertEqual((result["eval_set"], result["questions"], result["answerable"]), (DEV, 3, 2))
        self.assertEqual(set(result["retrievers"]), {"dense", "lexical", "hybrid"})
        code, out = self.run_main()  # an existing report is kept
        self.assertEqual(code, 1)
        self.assertIn("Use --force", out)

    def test_validation_set_is_refused_before_the_index_is_opened(self):
        copy_validation_set(self.root)
        code, out = self.run_main("--questions", guard.VALIDATION_SET)
        self.assertEqual(code, 2)
        self.assertIn("Refused", out)
        self.assertEqual((self.opened, self.reports()), ([], []))

    def test_invalid_question_set_writes_nothing(self):
        bad = {"questions": [{"id": "q1", "question": "x?", "relevant": [{"source": "docs/risk.md", "contains": "absent"}]}]}
        (self.root / "eval" / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
        code, out = self.run_main("--questions", "eval/bad.json")
        self.assertEqual(code, 1)
        self.assertIn("no indexed chunk contains its target phrase", out)
        self.assertEqual(self.reports(), [])


class TestGenerationEvaluationScript(ScriptCase):
    script = "evaluate_rag_generation"

    def test_validation_set_is_refused_before_ollama_is_contacted(self):
        copy_validation_set(self.root)
        with unittest.mock.patch.object(ans, "ollama_status", side_effect=AssertionError("contacted")):
            code, out = self.run_main("--questions", guard.VALIDATION_SET)
        self.assertEqual(code, 2)
        self.assertIn("must not be used to tune", out)
        self.assertEqual((self.opened, self.reports()), ([], []))

    def test_development_set_report_with_a_fake_local_model(self):
        def generator(question, evidence):
            return ans.INSUFFICIENT, {"model": "fake", "seconds": 0.0}
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": True, "version": "test"}), \
                unittest.mock.patch.object(ans, "ollama_generator", return_value=generator):
            code, out = self.run_main()
        self.assertEqual(code, 0, out)
        self.assertEqual(self.reports(), ["generation_eval.json", "generation_eval.md"])
        result = json.loads((self.root / "reports" / "generation_eval.json").read_text(encoding="utf-8"))
        self.assertEqual(result["eval_set"], DEV)
        self.assertEqual({r["outcome"] for r in result["results"]} - {"abstained_threshold"}, {"abstained_model"})

    def test_unavailable_ollama_writes_nothing(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "not running"}):
            code, out = self.run_main()
        self.assertEqual(code, 1)
        self.assertIn("not running", out)
        self.assertEqual(self.reports(), [])


class TestProjectFilesUnchanged(unittest.TestCase):
    def test_project_reports_are_not_written_by_these_tests(self):
        reports = PROJECT_ROOT / "data" / "reports" / "rag"
        if not reports.exists():
            self.skipTest("no RAG reports")
        before = {p.name: p.stat().st_mtime_ns for p in reports.iterdir()}
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRetrievalEvaluationScript)
        unittest.TextTestRunner(stream=io.StringIO()).run(suite)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in reports.iterdir()}, before)


if __name__ == "__main__":
    unittest.main()
