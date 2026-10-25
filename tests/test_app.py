"""Streamlit AppTest tests for the dashboard (docs/decisions.md, D21).

AppTest runs the page scripts in this process without starting a server or a
browser. Page tests on real artifacts are skipped if the frozen outputs are
absent; the RAG and workflow interaction tests also need the RAG index and the
cached embedding model. Missing-artifact tests point the dashboard at an empty
data folder.
"""

import json
import sys
import tempfile
import tomllib
import unittest
import unittest.mock
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from fincrisis.agents.graph import REFUSAL, build_graph
from fincrisis.agents.runtime import make_tools
from fincrisis.dashboard import disclosures
from fincrisis.dashboard import loaders as ld
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline
from tests.test_dashboard import DIRS, PRESENT, PROJECT_ROOT, tree_hashes
from tests.test_rag import MODEL, model_cached

APP = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP))  # as `streamlit run app/streamlit_app.py` does
import ui  # noqa: E402

PAGES = ["market_dashboard", "market_stress", "company_distress", "rag_assistant", "research_workflow"]
SETTINGS = ld.load_settings()
INDEX_READY = ld.rag_index_status(SETTINGS["rag"])["ready"] and model_cached(MODEL)


def run_page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "views" / f"{name}.py"), default_timeout=180).run()


def texts(at: AppTest, kind: str) -> list[str]:
    return [element.value for element in getattr(at, kind)]


class AppCase(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(setattr, ui, "DATA_ROOT", ui.DATA_ROOT)


class TestConfiguration(unittest.TestCase):
    def test_streamlit_is_local_only_without_usage_statistics(self):
        config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["server"]["address"], "localhost")
        self.assertIs(config["browser"]["gatherUsageStats"], False)
        self.assertIs(config["server"]["headless"], True)

    def test_pages_are_not_in_a_pages_folder(self):
        # A `pages/` folder next to the entry point would also be listed by Streamlit's automatic navigation.
        self.assertFalse((APP / "pages").exists())

    def test_local_only_problems(self):
        """Regression (review of 8f92b8d): a launch outside the project root served on all interfaces."""
        def options(address, stats):
            return unittest.mock.patch.object(ui.st, "get_option", side_effect=lambda name: {
                "server.address": address, "browser.gatherUsageStats": stats}[name])
        for address in ui.LOCAL_ADDRESSES:
            with options(address, False):
                self.assertEqual(ui.local_only_problems(), [])
        with options(None, True):  # what Streamlit uses when it does not find .streamlit/config.toml
            problems = ui.local_only_problems()
        self.assertEqual(len(problems), 2)
        self.assertIn("all network interfaces", problems[0])
        with options("0.0.0.0", False):
            self.assertIn("0.0.0.0", ui.local_only_problems()[0])

    def test_app_shows_nothing_unless_local_only(self):
        problems = ["the server address is 0.0.0.0, not localhost"]
        with unittest.mock.patch.object(ui, "local_only_problems", return_value=problems):
            for script in (APP / "streamlit_app.py", APP / "views" / "market_dashboard.py"):
                at = AppTest.from_file(str(script), default_timeout=60).run()
                self.assertFalse(at.exception)
                self.assertTrue(any("not running in local-only mode" in e.value for e in at.error))
                self.assertEqual(texts(at, "title"), [])
                self.assertEqual(len(at.get("plotly_chart")), 0)


@unittest.skipUnless(PRESENT, "frozen market/distress outputs not present")
class TestPagesOnFrozenOutputs(AppCase):
    def test_every_page_renders_with_notices_and_limitations(self):
        before = {k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}
        for name in PAGES:
            with self.subTest(page=name):
                at = run_page(name)
                self.assertFalse(at.exception, [e.message for e in at.exception])
                self.assertFalse(at.error, texts(at, "error"))
                notices = texts(at, "info") + texts(at, "warning")
                self.assertEqual(any("frozen snapshot as of" in n for n in notices), name != "market_dashboard",
                                 notices)
                shown = name != "market_dashboard"
                self.assertEqual(any(disclosures.RESEARCH_ONLY in c for c in texts(at, "caption")), shown)
                self.assertEqual("Limitations" in texts(at, "subheader"), shown)
        self.assertEqual({k: tree_hashes(PROJECT_ROOT / d) for k, d in DIRS.items()}, before)

    def test_market_dashboard_explains_each_indicator_in_red(self):
        at = run_page("market_dashboard")
        explanations = [c for c in texts(at, "caption") if "color: #a94442" in c]
        self.assertEqual(len(at.metric), 4)
        self.assertEqual(len(explanations), 4, texts(at, "caption"))
        self.assertIn("means it fell by 0.52% from the day before", explanations[0])

    def test_only_the_market_dashboard_drops_the_header_research_notice(self):
        header = f":material/info: {disclosures.RESEARCH_ONLY}"
        for name in PAGES:
            with self.subTest(page=name):
                captions = texts(run_page(name), "caption")
                self.assertEqual(header in captions, name != "market_dashboard", captions)

    def test_chart_pages_draw_their_charts(self):
        for name, minimum in (("market_dashboard", 2), ("market_stress", 6), ("company_distress", 5)):
            with self.subTest(page=name):
                self.assertGreaterEqual(len(run_page(name).get("plotly_chart")), minimum)

    def test_market_dashboard_charts_hide_the_zoom_and_pan_buttons(self):
        hidden = ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"]
        for name, expected in (("market_dashboard", hidden), ("market_stress", None)):
            with self.subTest(page=name):
                for element in run_page(name).get("plotly_chart"):
                    config = json.loads(element.proto.config or "{}")
                    self.assertEqual(config.get("modeBarButtonsToRemove"), expected)

    def test_distress_page_shows_the_label_caveat_and_no_scoring_form(self):
        at = run_page("company_distress")
        self.assertTrue(any("remain unverified" in w for w in texts(at, "warning")))
        self.assertEqual(len(at.text_input) + len(at.number_input), 0)

    def test_entry_point_opens_the_market_dashboard(self):
        at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=180).run()
        self.assertFalse(at.exception)
        self.assertEqual(texts(at, "title"), ["Market Dashboard"])


class TestMissingArtifacts(AppCase):
    def test_pages_explain_missing_artifacts_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ui.DATA_ROOT = Path(tmp)
            for name in PAGES:
                with self.subTest(page=name):
                    at = run_page(name)
                    self.assertFalse(at.exception, [e.message for e in at.exception])
                    if name in ("market_dashboard", "market_stress", "company_distress"):
                        self.assertTrue(at.error, f"{name} shows no error")
                        self.assertTrue(any("How to fix" in m for m in texts(at, "markdown")))
            rag = run_page("rag_assistant")
            self.assertTrue(any("cannot be used" in e for e in texts(rag, "error")))
            self.assertTrue(any("build_rag_index.py --force" in c for c in texts(rag, "code")))
            self.assertTrue(rag.button[0].disabled)  # no question can be sent to a missing index

    def test_altered_frozen_output_is_refused_on_the_page(self):
        if not PRESENT:
            self.skipTest("frozen outputs not present")
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            for d in DIRS.values():
                shutil.copytree(PROJECT_ROOT / d, Path(tmp) / d)
            path = Path(tmp) / DIRS["ms_reports"] / "test_predictions.csv"
            path.write_bytes(path.read_bytes().replace(b"2015-01-02", b"2015-01-01", 1))
            ui.DATA_ROOT = Path(tmp)
            at = run_page("market_stress")
            self.assertFalse(at.exception)
            self.assertTrue(any("test_predictions.csv does not match" in e for e in texts(at, "error")))


@unittest.skipUnless(PRESENT and INDEX_READY, "frozen outputs, RAG index or embedding model not present")
class TestInteractions(AppCase):
    def ask(self, page: str, key: str, question: str) -> AppTest:
        at = run_page(page)
        at.text_input(key=key).input(question)
        return next(b for b in at.button if b.label in ("Ask", "Run workflow")).click().run()

    def test_rag_page_gives_the_same_answer_as_the_pipeline(self):
        question = "Which VIX value is used for date t?"
        at = self.ask("rag_assistant", "rag_text", question)
        self.assertFalse(at.exception)
        rag = SETTINGS["rag"]
        store, _, embeddings = pipeline.open_store(rag)
        expected = ans.answer(question, store, embeddings, rag["top_k"], rag["min_relevance"],
                              rag["max_answer_sentences"], None, rag["ollama"]["min_support"],
                              pipeline.make_retriever(store, rag))
        markdown = texts(at, "markdown")
        self.assertIn(expected["answer"], markdown)
        self.assertTrue(any(f"`{expected['citations'][0]['source']}`" in m for m in markdown))

    def test_rag_page_abstains_without_evidence(self):
        at = self.ask("rag_assistant", "rag_text", "Who won the 2018 FIFA World Cup?")
        self.assertIn(ans.INSUFFICIENT, texts(at, "info"))

    def test_workflow_refuses_advice_and_reports_the_route(self):
        at = self.ask("research_workflow", "wf_text", "Should I sell my stocks now?")
        self.assertIn(REFUSAL, texts(at, "info"))
        self.assertTrue(any(c == "Route: route_request → refuse_advice" for c in texts(at, "caption")))

    def test_workflow_market_answer_matches_the_cli_wiring(self):
        question = "What is the current market stress level?"
        at = self.ask("research_workflow", "wf_text", question)
        tools, _ = make_tools(SETTINGS, use_llm=False)
        expected = build_graph(tools).invoke({"question": question})
        self.assertIn(expected["answer"], texts(at, "markdown"))
        self.assertTrue(any(c.endswith("retrieve_context → compose_answer") for c in texts(at, "caption")))


if __name__ == "__main__":
    unittest.main()
