"""Tests for the QA runner and the environment check (docs/decisions.md, D28).

The runner's test stage is tested with a stubbed subprocess, so the suite never
runs itself. The end-to-end script test uses --quick (no unit tests).
"""

import json
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from fincrisis.dashboard import loaders as ld
from fincrisis.qa import environment, runner
from fincrisis.qa.integrity import Check
from tests.test_integrity import ALL_PRESENT
from tests.test_rag import MODEL, model_cached

PROJECT_ROOT = Path(__file__).resolve().parents[1]
READY = ALL_PRESENT and ld.rag_index_status(ld.load_settings()["rag"])["ready"] and model_cached(MODEL)


class TestEnvironment(unittest.TestCase):
    def test_locked_versions_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "lock.txt"
            lock.write_text("# comment\n--extra-index-url x\nscikit_learn==1.9.1\ntorch==2.14.1+cpu\nGone==1.0\n",
                            encoding="utf-8")
            installed = {"scikit-learn": "1.9.2", "torch": "2.14.1+cpu", "extra-pkg": "0.1", "pip": "25"}
            checks = environment.checks(lock, installed)
        by_name = {c.name: c for c in checks}
        self.assertEqual(by_name["scikit-learn"].status, "mismatch")
        self.assertIn("installed 1.9.2, locked 1.9.1", by_name["scikit-learn"].detail)
        self.assertTrue(by_name["torch"].ok)
        self.assertEqual(by_name["gone"].status, "missing")
        self.assertEqual((by_name["unlocked packages"].status, by_name["unlocked packages"].detail), ("ok", "extra-pkg"))

    def test_project_environment_matches_the_lock(self):
        self.assertEqual([c for c in environment.checks() if not c.ok], [])


class TestParseUnittest(unittest.TestCase):
    def test_summaries(self):
        ok = runner.parse_unittest("....\n----\nRan 4 tests in 1.0s\n\nOK (skipped=2)\n", 0)
        self.assertEqual([(c.status, c.detail) for c in ok], [("ok", "4 tests (skipped=2)")])
        failed = runner.parse_unittest("FAIL: test_x (tests.test_a.TestA.test_x)\nRan 3 tests in 1s\n\nFAILED (failures=1)\n", 1)
        self.assertEqual([(c.name, c.status) for c in failed],
                         [("unittest", "mismatch"), ("tests.test_a.TestA.test_x", "mismatch")])
        self.assertEqual(runner.parse_unittest("Traceback: import error\n", 1)[0].status, "mismatch")
        self.assertEqual(runner.parse_unittest("Ran 1 test in 0s\n\nOK\n", 1)[0].status, "mismatch")  # exit code wins

    def test_live_switches_are_set_only_when_asked(self):
        seen = []

        def fake_run(args, **kwargs):
            seen.append({k: kwargs["env"].get(k) for k in runner.LIVE_SWITCHES})
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="Ran 1 test in 0s\n\nOK\n")
        with unittest.mock.patch.dict("os.environ", {"FINCRISIS_LIVE_OLLAMA": "1"}), \
                unittest.mock.patch.object(runner.subprocess, "run", fake_run):
            self.assertTrue(runner.run_tests(live=False)[0].ok)
            runner.run_tests(live=True)
        self.assertEqual(seen, [{"FINCRISIS_LIVE_OLLAMA": None, "FINCRISIS_LIVE_SERVER": None},
                                {"FINCRISIS_LIVE_OLLAMA": "1", "FINCRISIS_LIVE_SERVER": "1"}])


class TestRunAndReport(unittest.TestCase):
    def test_a_failing_or_crashing_stage_fails_the_run_without_hiding_others(self):
        def crash():
            raise RuntimeError("boom")
        stages = runner.run([("good", lambda: [Check("g", "a", "ok")]),
                             ("bad", lambda: [Check("b", "x.json", "mismatch", "sha differs")]),
                             ("crash", crash), ("empty", lambda: []), ("after", lambda: [Check("g", "b", "ok")])])
        self.assertEqual([s.ok for s in stages], [True, False, False, False, True])
        text = runner.report(stages)
        self.assertIn("PASS  good", text)
        self.assertIn("MISMATCH: b / x.json (sha differs)", text)
        self.assertIn("RuntimeError: boom", text)
        self.assertTrue(text.endswith("QA failed."))
        self.assertTrue(runner.report(runner.run([("good", lambda: [Check("g", "a", "ok")])])).endswith("QA passed."))


class TestScript(unittest.TestCase):
    def script(self, *args):
        return subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "run_qa.py"), *args],
                              capture_output=True, text=True, cwd=PROJECT_ROOT)

    def test_quick_and_live_cannot_be_combined(self):
        out = self.script("--quick", "--live")
        self.assertEqual(out.returncode, 2)
        self.assertIn("--live applies to the test suite", out.stderr)

    @unittest.skipUnless(READY, "frozen artifacts, current RAG index or embedding model not present")
    def test_quick_run_on_the_project(self):
        out = self.script("--quick", "--json")
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])
        stages = json.loads(out.stdout)
        self.assertEqual([s["stage"] for s in stages], ["environment", "integrity", "reproducibility", "rag regression"])
        self.assertTrue(all(s["ok"] for s in stages))


if __name__ == "__main__":
    unittest.main()
