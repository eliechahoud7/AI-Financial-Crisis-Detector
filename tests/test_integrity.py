"""Tests for the integrity verifier (docs/decisions.md, D22).

Tamper tests run on temporary copies of the frozen inputs and outputs (skipped
if they are absent), so the project's files are never touched.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fincrisis.dashboard import loaders as ld
from fincrisis.qa import integrity
from tests.test_dashboard import DIRS, PATHS, PRESENT, PROJECT_ROOT
from tests.test_rag import build, write_corpus

SETTINGS = ld.load_settings()
VALIDATION = "knowledge_base/eval/validation_questions.json"
REPORT = (Path(PATHS["data_reports"]) / "market_stress" / "dataset_report.md").as_posix()
TAIWAN = [Path(PATHS["raw_data"]) / "taiwan_bankruptcy", Path(PATHS["interim_data"]) / "taiwan_bankruptcy"]
ALL_PRESENT = PRESENT and all((PROJECT_ROOT / d).exists() for d in TAIWAN)


def failures(checks, group=None):
    return {(c.group, c.name, c.status) for c in checks if not c.ok and (group is None or c.group == group)}


@unittest.skipUnless(ALL_PRESENT, "frozen inputs and outputs not present")
class TestVerifierOnCopies(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for d in [*DIRS.values(), *TAIWAN]:
            shutil.copytree(PROJECT_ROOT / d, self.root / d)
        for name in (VALIDATION, REPORT):  # manifest entries outside the copied directories
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / name, self.root / name)

    def tearDown(self):
        self.tmp.cleanup()

    def frozen(self):
        """Every check except the RAG index, which these copies do not include."""
        return [c for c in integrity.verify_all(self.root, SETTINGS) if c.group != "rag index"]

    def test_intact_copies_pass(self):
        checks = self.frozen()
        self.assertEqual(failures(checks), set())
        self.assertEqual({c.group for c in checks}, {"manifest", "market snapshot", "distress data"})
        self.assertIn(VALIDATION, {c.name for c in checks})

    def test_altered_raw_snapshot_is_caught_even_with_matching_metadata(self):
        raw = self.root / DIRS["snapshot"] / "gspc.csv"
        raw.write_bytes(raw.read_bytes() + b"\n")
        self.assertEqual(failures(self.frozen()), {
            ("market snapshot", "gspc.csv (frozen selection record)", "mismatch"),
            ("market snapshot", "gspc.csv (snapshot metadata)", "mismatch")})
        meta_path = self.root / DIRS["snapshot"] / "metadata.json"  # someone also updates the snapshot's own record
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["series"]["gspc"]["sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        self.assertEqual(failures(self.frozen()), {
            ("market snapshot", "gspc.csv (frozen selection record)", "mismatch")})

    def test_altered_or_missing_distress_data_is_caught(self):
        raw = self.root / TAIWAN[0] / "data.csv"
        raw.write_bytes(raw.read_bytes().replace(b"\n", b"\r\n", 1))
        (self.root / TAIWAN[1] / "split.csv").unlink()
        self.assertEqual(failures(self.frozen(), "distress data"), {
            ("distress data", "raw data.csv", "mismatch"), ("distress data", "processed split.csv", "missing")})

    def test_altered_validation_questions_are_caught(self):
        path = self.root / VALIDATION
        questions = json.loads(path.read_text(encoding="utf-8"))
        questions["questions"][0]["question"] += " (edited)"
        path.write_text(json.dumps(questions), encoding="utf-8")
        self.assertEqual(failures(self.frozen()), {("manifest", VALIDATION, "mismatch")})

    def test_unreadable_anchor_is_reported_not_raised(self):
        (self.root / DIRS["ms_reports"] / "selection.json").unlink()
        (self.root / DIRS["tw_reports"] / "test_metrics.json").write_text("{not json", encoding="utf-8")
        found = failures(self.frozen())
        self.assertIn(("market snapshot", "selection.json", "missing"), found)
        self.assertIn(("distress data", "test_metrics.json", "missing"), found)


class TestRagIndexCheck(unittest.TestCase):
    RAG = {"sources": ["docs/*.md", "docs/*.txt"], "index_dir": "index", "embedding_model": "fake-bow",
           "retrieval": {"embed_section_headings": False}}

    def test_ready_missing_and_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            self.assertEqual(integrity.rag_index_checks(root, self.RAG)[0].status, "missing")
            build(root)
            self.assertTrue(integrity.rag_index_checks(root, self.RAG)[0].ok)
            (root / "docs" / "notes.txt").write_text("Changed.\n", encoding="utf-8")
            stale = integrity.rag_index_checks(root, self.RAG)[0]
            self.assertEqual(stale.status, "mismatch")
            self.assertIn("build_rag_index.py --force", stale.detail)


class TestValidationSetProvenance(unittest.TestCase):
    def test_recorded_checksum_is_the_committed_file(self):
        recorded = ld.frozen_checksums()[VALIDATION]
        group = next(g for g in json.loads(ld.FROZEN_MANIFEST.read_text(encoding="utf-8"))["provenance"]
                     if VALIDATION in g["files"])
        commit = group["committed"].split(",")[0]
        try:
            blob = subprocess.run(["git", "show", f"{commit}:{VALIDATION}"], cwd=PROJECT_ROOT, capture_output=True,
                                  check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            self.skipTest("git history not available")
        self.assertEqual(hashlib.sha256(blob).hexdigest(), recorded)

    def test_checksummed_text_files_keep_their_bytes_in_every_checkout(self):
        out = subprocess.run(["git", "check-attr", "eol", "--", VALIDATION, "config/frozen_artifacts.json"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            self.skipTest("git not available")
        self.assertEqual(out.stdout.count("eol: lf"), 2, out.stdout)

    def test_validation_file_matches_in_this_checkout(self):
        self.assertTrue(integrity._compare("manifest", VALIDATION, PROJECT_ROOT / VALIDATION,
                                           ld.frozen_checksums()[VALIDATION]).ok)


class TestScript(unittest.TestCase):
    @unittest.skipUnless(ALL_PRESENT, "frozen inputs and outputs not present")
    def test_cli_on_the_project(self):
        out = subprocess.run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "verify_integrity.py"), "--json"],
                             capture_output=True, text=True, cwd=PROJECT_ROOT)
        checks = json.loads(out.stdout)
        failed = [c for c in checks if c["status"] != "ok"]
        self.assertEqual(out.returncode, 1 if failed else 0)
        self.assertEqual([c for c in failed if c["group"] != "rag index"], [])  # the index may be stale mid-edit
        self.assertGreaterEqual(len(checks), 36)

    def test_summary_lists_failures(self):
        checks = [integrity.Check("manifest", "a.json", "ok"), integrity.Check("manifest", "b.json", "mismatch", "x")]
        text = integrity.summary(checks)
        self.assertIn("manifest: 1/2 ok", text)
        self.assertIn("MISMATCH: b.json (x)", text)
        self.assertIn("1 integrity check(s) failed.", text)


if __name__ == "__main__":
    unittest.main()
