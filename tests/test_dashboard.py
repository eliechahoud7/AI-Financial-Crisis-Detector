"""Tests for the dashboard's shared loaders and disclosures (docs/decisions.md, D18).

Loader tests run on temporary copies of the frozen outputs (skipped if they
are absent), so artifacts can be removed or altered safely. They check that
loaders verify integrity, report missing or altered artifacts as
`ArtifactError` with a remedy, never write, and never fit a model.
Disclosure tests use small hand-made inputs.
"""

import hashlib
import json
import shutil
import tempfile
import unittest
import unittest.mock
from datetime import date
from pathlib import Path

import yaml
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fincrisis.agents import runtime, tools
from fincrisis.agents.graph import build_graph
from fincrisis.dashboard import disclosures as dis
from fincrisis.dashboard import loaders as ld
from tests.test_agent import real_tools
from tests.test_rag import build, write_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((PROJECT_ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
PATHS = SETTINGS["paths"]
DIRS = {
    "snapshot": Path(PATHS["raw_data"]) / "market",
    "dataset": Path(PATHS["processed_data"]) / "market_stress",
    "ms_reports": Path(PATHS["data_reports"]) / "market_stress" / "models",
    "ms_models": Path(PATHS["models"]) / "market_stress",
    "tw_reports": Path(PATHS["data_reports"]) / "taiwan_bankruptcy" / "baselines",
    "tw_models": Path(PATHS["models"]) / "taiwan_bankruptcy",
}
PRESENT = all((PROJECT_ROOT / d).exists() for d in DIRS.values())


def tree_hashes(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@unittest.skipUnless(PRESENT, "frozen market/distress outputs not present")
class TestLoaders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for d in DIRS.values():
            shutil.copytree(PROJECT_ROOT / d, self.root / d)

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, key: str, name: str) -> Path:
        return self.root / DIRS[key] / name

    def assert_artifact_error(self, load, *fragments):
        with self.assertRaises(ld.ArtifactError) as caught:
            load()
        for fragment in fragments:
            self.assertIn(fragment, str(caught.exception))
        self.assertTrue(caught.exception.remedy)
        return caught.exception

    def test_loaders_read_verified_outputs_without_writing_or_fitting(self):
        before = tree_hashes(self.root)
        with unittest.mock.patch.object(LogisticRegression, "fit", side_effect=AssertionError("fit called")), \
                unittest.mock.patch.object(XGBClassifier, "fit", side_effect=AssertionError("fit called")):
            snapshot = ld.market_snapshot(self.root, PATHS)
            features, _ = ld.market_features(snapshot)
            dataset, _ = ld.market_dataset(self.root, PATHS)
            stress = ld.market_stress_results(self.root, PATHS)
            signal = ld.current_market_signal(self.root, PATHS, 7, today=snapshot["as_of"])
            distress = ld.distress_results(self.root, PATHS)
        self.assertEqual(tree_hashes(self.root), before)
        self.assertEqual(features.index[-1].date(), snapshot["as_of"])
        self.assertEqual(signal["as_of"], str(snapshot["as_of"]))
        self.assertEqual(stress["primary_model"], stress["selection"]["primary_model"]["name"])
        self.assertEqual(stress["thresholds"][stress["primary_model"]], signal["primary"]["threshold"])
        self.assertEqual(len(stress["test_predictions"]), stress["test_metrics"]["rows"])
        self.assertLess(dataset.index[-1].date(), snapshot["as_of"])  # the last 20 sessions have no label
        self.assertIn("unverified", distress["label_caveat"])

    def test_missing_artifacts_are_reported_with_a_remedy(self):
        empty = self.root / "empty"
        empty.mkdir()
        for load in (lambda: ld.market_snapshot(empty, PATHS), lambda: ld.market_dataset(empty, PATHS),
                     lambda: ld.market_stress_results(empty, PATHS), lambda: ld.distress_results(empty, PATHS),
                     lambda: ld.current_market_signal(empty, PATHS, 7)):
            self.assert_artifact_error(load)
        error = self.assert_artifact_error(lambda: ld.market_snapshot(empty, PATHS))
        self.assertIn("download_market_data.py would create a new snapshot", error.remedy)

    def test_altered_snapshot_is_refused(self):
        raw = next(self.path("snapshot", "").glob("*.csv"))
        raw.write_bytes(raw.read_bytes() + b"\n")
        self.assert_artifact_error(lambda: ld.market_snapshot(self.root, PATHS), "recorded checksum")

    def edit_cell(self, path: Path, row: int, column: int, value: str) -> None:
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        cells = lines[row].rstrip("\n").split(",")
        cells[column] = value
        lines[row] = ",".join(cells) + "\n"
        path.write_text("".join(lines), encoding="utf-8")

    def test_altered_market_stress_outputs_are_refused(self):
        load = lambda: ld.market_stress_results(self.root, PATHS)  # noqa: E731
        selection = self.path("ms_reports", "selection.json")
        original = selection.read_bytes()
        selection.write_bytes(original + b" ")
        self.assert_artifact_error(load, "selection.json does not match its checksum in config/frozen_artifacts.json")
        with unittest.mock.patch.object(ld, "_verify_frozen"):  # the pipeline's own record still catches it
            self.assert_artifact_error(load, "selection.json differs")
        selection.write_bytes(original)

        model = self.path("ms_models", "xgboost.joblib")
        saved = model.read_bytes()
        model.write_bytes(b"tampered")
        self.assert_artifact_error(load, "xgboost.joblib does not match its checksum")
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            self.assert_artifact_error(load, "xgboost.joblib is missing or does not match its recorded checksum")
        model.write_bytes(saved)

        predictions = self.path("ms_reports", "test_predictions.csv")
        saved = predictions.read_bytes()
        self.edit_cell(predictions, 1, -1, "0.999")  # one changed xgboost score
        self.assert_artifact_error(load, "test_predictions.csv does not match its checksum")
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            self.assert_artifact_error(load, "average precision of xgboost")
        predictions.write_bytes(saved)

    def test_score_preserving_prediction_edits_are_refused(self):
        """Regression: edits that keep every recorded metric were not detected before the manifest (D19)."""
        load = lambda: ld.market_stress_results(self.root, PATHS)  # noqa: E731
        predictions = self.path("ms_reports", "test_predictions.csv")
        self.edit_cell(predictions, 1, 0, "2014-12-31")  # a changed date: no metric changes
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            load()  # the metric recomputation alone cannot see it
        self.assert_artifact_error(load, "test_predictions.csv does not match its checksum")

        distress = lambda: ld.distress_results(self.root, PATHS)  # noqa: E731
        predictions = self.path("tw_reports", "test_predictions.csv")
        self.edit_cell(predictions, 1, 0, "999999")  # a changed row_id: no metric changes
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            distress()
        self.assert_artifact_error(distress, "test_predictions.csv does not match its checksum")

    def test_altered_metrics_and_reports_are_refused(self):
        for key, name in (("ms_reports", "test_metrics.json"), ("ms_reports", "final_report.md"),
                          ("tw_reports", "test_metrics.json"), ("tw_reports", "cv_results.csv")):
            path = self.path(key, name)
            saved = path.read_bytes()
            path.write_bytes(saved + b"\n")
            load = ld.market_stress_results if key == "ms_reports" else ld.distress_results
            self.assert_artifact_error(lambda: load(self.root, PATHS), f"{name} does not match its checksum")
            path.write_bytes(saved)

    def test_altered_dataset_is_refused_by_the_market_signal(self):
        """Regression: assess_market read dataset.csv without checking it (D19)."""
        self.edit_cell(self.path("dataset", "dataset.csv"), 300, 1, "0")  # one changed value
        with self.assertRaises(tools.ToolError) as caught:
            tools.assess_market(self.root, PATHS, 7, today=date(2026, 10, 24))
        self.assertIn("dataset.csv differs from the dataset the frozen evaluation used", str(caught.exception))
        self.assert_artifact_error(lambda: ld.current_market_signal(self.root, PATHS, 7), "dataset.csv")
        self.assert_artifact_error(lambda: ld.market_dataset(self.root, PATHS), "dataset.csv")
        state = build_graph(real_tools(self.root)).invoke({"question": "What is the current market stress level?"})
        self.assertEqual(state["trace"][-1], "report_error")
        self.assertIn("dataset.csv differs", state["answer"])

    def test_missing_dataset_keeps_the_previous_market_signal_behaviour(self):
        self.path("dataset", "dataset.csv").unlink()
        signal = tools.assess_market(self.root, PATHS, 7, today=date(2026, 10, 24))  # as before: no percentile
        self.assertNotIn("historical_percentile", signal["primary"])
        self.assert_artifact_error(lambda: ld.current_market_signal(self.root, PATHS, 7), "missing")

    def test_manifest_is_read_from_the_checkout_not_from_the_data_root(self):
        bogus = self.root / "config" / "frozen_artifacts.json"
        bogus.parent.mkdir()
        bogus.write_text(json.dumps({"files": {}}), encoding="utf-8")  # a data copy cannot vouch for itself
        self.edit_cell(self.path("ms_reports", "test_predictions.csv"), 1, 0, "2014-12-31")
        self.assert_artifact_error(lambda: ld.market_stress_results(self.root, PATHS), "does not match its checksum")

    def test_missing_manifest_is_reported(self):
        with unittest.mock.patch.object(ld, "FROZEN_MANIFEST", self.root / "nowhere.json"):
            error = self.assert_artifact_error(lambda: ld.market_stress_results(self.root, PATHS), "nowhere.json")
        self.assertIn("frozen_artifacts.json from Git", error.remedy)

    def test_altered_distress_outputs_are_refused(self):
        load = lambda: ld.distress_results(self.root, PATHS)  # noqa: E731
        meta = json.loads(self.path("tw_models", "model_metadata.json").read_text(encoding="utf-8"))
        model = self.path("tw_models", meta["models"]["xgb_large_as_missing"]["file"])
        saved = model.read_bytes()
        model.write_bytes(b"tampered")
        self.assert_artifact_error(load, "does not match its checksum")
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            self.assert_artifact_error(load, "does not match its recorded checksum")
        model.write_bytes(saved)

        self.edit_cell(self.path("tw_reports", "test_predictions.csv"), 1, 3, "0.999")  # one changed lr_quantile score
        self.assert_artifact_error(load, "test_predictions.csv does not match its checksum")
        with unittest.mock.patch.object(ld, "_verify_frozen"):
            self.assert_artifact_error(load, "average precision of logistic_regression")

    def test_frozen_project_outputs_are_unchanged_by_loading(self):
        before = {k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}
        ld.market_stress_results(PROJECT_ROOT, PATHS)
        ld.distress_results(PROJECT_ROOT, PATHS)
        ld.market_snapshot(PROJECT_ROOT, PATHS)
        self.assertEqual({k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}, before)


class TestFrozenManifest(unittest.TestCase):
    def test_manifest_lists_only_frozen_outputs_with_recorded_provenance(self):
        manifest = json.loads(ld.FROZEN_MANIFEST.read_text(encoding="utf-8"))
        allowed = tuple(d.as_posix() + "/" for k, d in DIRS.items() if k != "snapshot") + (
            (Path(PATHS["data_reports"]) / "market_stress" / "dataset_report.md").as_posix(),)
        self.assertTrue(all(path.startswith(allowed) for path in manifest["files"]), manifest["files"])
        listed = {f for group in manifest["provenance"] for f in group["files"]}
        self.assertEqual(listed, set(manifest["files"]))
        for group in manifest["provenance"]:
            self.assertLess(group["outputs_written"], group["hashes_recorded"])

    @unittest.skipUnless(PRESENT, "frozen market/distress outputs not present")
    def test_project_outputs_match_the_manifest(self):
        for path, digest in ld.frozen_checksums().items():
            self.assertEqual(hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest(), digest, path)


class TestRagIndexStatus(unittest.TestCase):
    RAG = {"sources": ["docs/*.md", "docs/*.txt"], "index_dir": "index", "embedding_model": "fake-bow",
           "retrieval": {"embed_section_headings": False}}

    def test_ready_missing_altered_stale_and_wrong_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_corpus(root)
            self.assertFalse(ld.rag_index_status(self.RAG, root)["ready"])  # no index yet
            build(root)
            status = ld.rag_index_status(self.RAG, root)
            self.assertTrue(status["ready"], status)
            self.assertIsNone(status["remedy"])
            other = ld.rag_index_status({**self.RAG, "embedding_model": "other-model"}, root)
            self.assertIn("built with fake-bow", other["problem"])
            (root / "docs" / "notes.txt").write_text("Changed after the build.\n", encoding="utf-8")
            stale = ld.rag_index_status(self.RAG, root)
            self.assertIn("out of date for docs/notes.txt", stale["problem"])
            self.assertIn("build_rag_index.py --force", stale["remedy"])
            (root / "index" / "chunks.json").write_text("[]", encoding="utf-8")
            self.assertIn("chunks.json does not match", ld.rag_index_status(self.RAG, root)["problem"])

    def test_status_does_not_load_the_embedding_model(self):
        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(ld.pipeline, "make_embeddings", side_effect=AssertionError("loaded")):
            write_corpus(Path(tmp))
            build(Path(tmp))
            self.assertTrue(ld.rag_index_status(self.RAG, Path(tmp))["ready"])


SNAPSHOT = {"as_of": date(2026, 10, 23), "downloaded_at_utc": "2026-10-24T14:29:01+00:00", "source": "Test source"}


def stress_results(precision=0.33):
    return {
        "primary_model": "rule_x",
        "selection": {"caveat": "Scores rank risk (D2).", "primary_model": {"name": "rule_x", "rule": "best on validation"}},
        "test_metrics": {
            "prevalence": 0.212, "test": {"first_date": "2015-01-02", "last_date": "2026-09-25"},
            "bootstrap": {"primary": {"average_precision": {"rule_x": [0.25, 0.47]}}},
            "models": {"rule_x": {"average_precision": 0.363,
                                  "at_max_f1_threshold": {"precision": precision, "recall": 0.68}}}},
    }


class TestDisclosures(unittest.TestCase):
    def test_snapshot_notice_states_date_and_warns_when_stale(self):
        fresh = dis.snapshot_notice(SNAPSHOT, 7, today=date(2026, 10, 30))  # 7 days: not yet stale
        self.assertEqual(fresh.level, "info")
        self.assertIn("as of 2026-10-23", fresh.text)
        self.assertIn("never downloads", fresh.text)
        stale = dis.snapshot_notice(SNAPSHOT, 7, today=date(2026, 10, 31))
        self.assertEqual(stale.level, "warning")
        self.assertIn("8 days old", stale.text)

    def test_market_limitations_use_the_recorded_numbers(self):
        text = " ".join(dis.market_stress_limitations(stress_results()))
        for fragment in ("Scores rank risk (D2).", "`rule_x`", "best on validation", "0.363", "0.250 to 0.470",
                         "base rate of 0.212", "precision is 0.33", "most alerts are false alarms", "need not hold"):
            self.assertIn(fragment, text)
        self.assertIn("many alerts are false alarms", " ".join(dis.market_stress_limitations(stress_results(0.6))))

    def test_distress_limitations_flag_indistinguishable_models(self):
        results = {"test_metrics": {"label_caveat": "Labels unverified.", "paired_average_precision_differences": {
            "lr - dummy": {"interval": [-0.1, 0.2]},  # comparisons with the dummy are not limitations
            "xgb - lr": {"interval": [-0.12, 0.16]},
            "xgb - other": {"interval": [0.05, 0.2]}}}}
        lines = dis.distress_limitations(results)
        self.assertEqual(lines[0], "Labels unverified.")
        flagged = [line for line in lines if "does not distinguish" in line]
        self.assertEqual(len(flagged), 1)
        self.assertIn("`xgb - lr`", flagged[0])
        self.assertTrue(any("not calibrated" in line for line in lines))

    def test_rag_limitations_mention_threshold_and_model_licence(self):
        rag = {"min_relevance": 0.3, "ollama": {"model": "qwen2.5:3b"}}
        text = " ".join(dis.rag_limitations(rag))
        self.assertIn("0.30", text)
        self.assertIn("non-commercial", text)
        self.assertNotIn("non-commercial", " ".join(dis.rag_limitations({**rag, "ollama": {"model": "other"}})))

    def test_research_only_matches_the_workflow_disclaimer(self):
        from fincrisis.agents.graph import DISCLAIMER
        self.assertEqual(dis.RESEARCH_ONLY, DISCLAIMER)
        error = ld.ArtifactError("missing x", "Restore x.")
        self.assertEqual(dis.artifact_notice(error), dis.Notice("error", "missing x Restore x."))


class TestRuntime(unittest.TestCase):
    def test_cli_uses_the_shared_runtime(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_crisis_agent", PROJECT_ROOT / "scripts" / "run_crisis_agent.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        self.assertIs(script.make_tools, runtime.make_tools)

    def test_tools_do_not_load_the_index_until_documents_are_needed(self):
        with unittest.mock.patch.object(runtime.pipeline, "open_store", side_effect=AssertionError("index loaded")):
            tools, notes = runtime.make_tools(SETTINGS, use_llm=True)
        self.assertEqual(notes, [])
        self.assertTrue(callable(tools.rag_answer) and callable(tools.context_answer))


if __name__ == "__main__":
    unittest.main()
