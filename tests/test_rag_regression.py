"""Tests for the RAG regression checks (docs/decisions.md, D25).

Comparison logic is tested on small hand-made results. The check on the real
index runs only when the index is current and the embedding model is cached.
The validation set is never evaluated; one test confirms it is refused.
"""

import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from fincrisis.dashboard import loaders as ld
from fincrisis.qa import rag_regression as rr
from fincrisis.qa import validation_guard as guard
from tests.test_rag import MODEL, model_cached

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ld.load_settings()
READY = ld.rag_index_status(SETTINGS["rag"])["ready"] and model_cached(MODEL)


def mode(hit1, mrr, ranks):
    return {"summary": {"hit@1": hit1, "mrr": mrr}, "ranks": ranks}


class TestCompareRetrieval(unittest.TestCase):
    BASELINE = {"modes": {"dense": mode(0.5, 0.6, {"q1": 1, "q2": 3}), "lexical": mode(0.5, 0.6, {"q1": 1, "q2": 2})}}

    def test_configured_mode_may_not_get_worse(self):
        current = {"dense": mode(0.5, 0.55, {"q1": 1, "q2": 4}), "lexical": mode(0.5, 0.6, {"q1": 1, "q2": 2})}
        bad = [c for c in rr.compare_retrieval(current, self.BASELINE, "dense") if not c.ok]
        self.assertEqual([(c.group, c.name) for c in bad], [("rag retrieval (dense, configured)", "mrr")])
        self.assertIn("('q2', 3, 4)", bad[0].detail)

    def test_other_modes_and_improvements_pass(self):
        current = {"dense": mode(0.6, 0.7, {"q1": 1, "q2": 1}), "lexical": mode(0.0, 0.1, {"q1": None, "q2": None})}
        self.assertTrue(all(c.ok for c in rr.compare_retrieval(current, self.BASELINE, "dense")))


class TestCompareAnswers(unittest.TestCase):
    RECORD = {"route": "documents", "trace": ["route_request", "retrieve_context", "compose_answer"],
              "answer": "A [1]", "citations": [{"marker": 1, "source": "a.md", "section": "S"}], "warnings": []}

    def test_any_change_is_reported_with_its_fields(self):
        baseline = {"answers": {"answer": {}, "workflow": {"q": self.RECORD}}}
        changed = {**self.RECORD, "answer": "B [1]", "citations": []}
        checks = rr.compare_answers({"answer": {"new question": {}}, "workflow": {"q": changed}}, baseline)
        self.assertEqual([(c.name, c.status) for c in checks], [("new question", "missing"), ("q", "mismatch")])
        self.assertEqual(checks[1].detail, "changed: answer, citations")
        self.assertTrue(rr.compare_answers({"answer": {}, "workflow": {"q": self.RECORD}}, baseline)[0].ok)

    def test_only_date_dependent_warnings_are_ignored(self):
        state = {"route": "market", "trace": [], "answer": "x", "citations": [],
                 "warnings": ["market snapshot is 30 days old (as of 2026-10-23); refresh it", "index is stale"]}
        self.assertEqual(rr._workflow_record(state)["warnings"], ["index is stale"])


class TestDevelopmentSetOnly(unittest.TestCase):
    def test_validation_set_cannot_be_the_development_set(self):
        rag = {**SETTINGS["rag"], "eval_set": guard.VALIDATION_SET}
        with self.assertRaises(guard.QuestionSetError):
            rr.development_questions(rag)

    def test_fixed_questions_come_from_the_development_set_or_the_readme(self):
        development = {q["question"] for q in rr.development_questions(SETTINGS["rag"])}
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        for question in rr.ANSWER_QUESTIONS + rr.WORKFLOW_QUESTIONS:
            self.assertTrue(question in development or question in readme, question)


class TestUpdate(unittest.TestCase):
    def test_update_writes_only_the_baselines_and_keeps_the_d16_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            qa = Path(tmp)
            (qa / "rag_dev_baseline.json").write_text(json.dumps({"reference_d16": {"kept": True}, "modes": {}}))
            fake = {"dense": mode(0.5, 0.6, {"q1": 1})}
            with unittest.mock.patch.multiple(rr, QA_DIR=qa, RETRIEVAL_BASELINE=qa / "rag_dev_baseline.json",
                                              ANSWER_BASELINE=qa / "reference_answers.json",
                                              retrieval=lambda *a: fake, development_questions=lambda *a: [],
                                              reference_answers=lambda *a: {"answer": {}, "workflow": {}}), \
                    unittest.mock.patch.object(rr.loaders, "rag_index_status",
                                               return_value={"ready": True, "built_at_utc": "t", "chunks": 1}), \
                    unittest.mock.patch.object(rr.pipeline, "open_store", return_value=(None, {}, None)):
                checks = rr.run(SETTINGS, update=True)
            self.assertTrue(checks[0].ok)
            self.assertEqual(sorted(p.name for p in qa.iterdir()), ["rag_dev_baseline.json", "reference_answers.json"])
            written = json.loads((qa / "rag_dev_baseline.json").read_text(encoding="utf-8"))
            self.assertEqual((written["reference_d16"], written["modes"]), ({"kept": True}, fake))

    def test_stale_index_is_reported_before_anything_runs(self):
        with unittest.mock.patch.object(rr.loaders, "rag_index_status",
                                        return_value={"ready": False, "problem": "out of date", "remedy": "rebuild"}), \
                unittest.mock.patch.object(rr.pipeline, "open_store", side_effect=AssertionError("opened")):
            checks = rr.run(SETTINGS)
        self.assertEqual([(c.name, c.status) for c in checks], [("index", "missing")])


@unittest.skipUnless(READY, "RAG index not current or embedding model not cached")
class TestProject(unittest.TestCase):
    def test_baselines_hold_and_record_the_d16_reference(self):
        checks = rr.run(SETTINGS)
        self.assertEqual([c for c in checks if not c.ok], [])
        baseline = json.loads(rr.RETRIEVAL_BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(baseline["eval_set"], SETTINGS["rag"]["eval_set"])
        self.assertEqual(baseline["reference_d16"]["eval_set"], SETTINGS["rag"]["eval_set"])

    def test_cli(self):
        out = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "check_rag_regression.py"), "--json"],
                             capture_output=True, text=True, cwd=PROJECT_ROOT)
        self.assertEqual(out.returncode, 0, out.stdout[-1500:])
        self.assertTrue(all(c["status"] == "ok" for c in json.loads(out.stdout)))


if __name__ == "__main__":
    unittest.main()
