"""Ollama failure handling across the RAG, the workflow and the dashboard (docs/decisions.md, D26).

A small stand-in HTTP server on 127.0.0.1 plays Ollama, so the real network
code (`urllib`) runs against slow, failing, malformed and empty replies without
a model. Workflow and dashboard tests inject the same failures. The live test
against the real local model runs only when FINCRISIS_LIVE_OLLAMA=1 is set and
the server has the configured model; it checks structure, not wording.
"""

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import streamlit as st

from fincrisis.agents import runtime
from fincrisis.agents.graph import build_graph
from fincrisis.dashboard import loaders as ld
from fincrisis.rag import answer as ans
from fincrisis.rag import pipeline
from tests.test_rag import MODEL, build, model_cached, write_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ld.load_settings()
QUESTION = "What is liquidity risk?"
GOOD = {"response": "Liquidity risk is the danger that a firm cannot meet short-term obligations [1].",
        "prompt_eval_count": 10, "eval_count": 5, "total_duration": 1_000_000_000}


class FakeOllama:
    """Serves /api/version, /api/tags and /api/generate; `behaviour` chooses how /api/generate replies."""

    def __init__(self):
        self.behaviour, self.requests = "good", []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code: int, body: bytes):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except ConnectionError:  # the client gave up (the "slow" case): nothing to answer
                    pass

            def do_GET(self):
                if self.path == "/api/version":
                    self.reply(200, b'{"version": "test"}')
                elif self.path == "/api/tags":
                    tags = b"{not json" if outer.behaviour == "bad_tags" else b'{"models": [{"name": "m"}]}'
                    self.reply(200, tags)
                else:
                    self.reply(404, b"{}")

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                behaviour = outer.behaviour
                if behaviour == "slow":
                    time.sleep(1.5)
                    self.reply(200, json.dumps(GOOD).encode())
                elif behaviour == "error":
                    self.reply(500, b'{"error": "model crashed"}')
                elif behaviour == "malformed":
                    self.reply(200, b'{"response": "trunc')
                elif behaviour == "empty":
                    self.reply(200, b'{"response": ""}')
                else:
                    self.reply(200, json.dumps(GOOD).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class TestRealTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ollama = FakeOllama()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_corpus(root)
        cls.store = build(root)[0]

    @classmethod
    def tearDownClass(cls):
        cls.ollama.close()
        cls.tmp.cleanup()

    def ask(self, behaviour: str, timeout: float = 5.0) -> dict:
        self.ollama.behaviour = behaviour
        generator = ans.ollama_generator("m", self.ollama.url, timeout=timeout)
        return ans.answer(QUESTION, self.store, self.store.embeddings, 3, 0.1, 2, generator)

    def test_good_reply_is_used_when_grounded(self):
        result = self.ask("good")
        self.assertEqual((result["status"], result["method"]), ("answered", "generative"))
        self.assertEqual(result["generation"]["seconds"], 1.0)
        self.assertFalse(self.ollama.requests[-1]["stream"])

    def test_failures_fall_back_to_the_extractive_answer_with_a_warning(self):
        extractive = ans.answer(QUESTION, self.store, self.store.embeddings, 3, 0.1, 2)["answer"]
        for behaviour, timeout, expected in (("error", 5.0, "generator failed (HTTPError"),
                                             ("malformed", 5.0, "generator failed (JSONDecodeError"),
                                             ("slow", 0.3, "generator failed ("),
                                             ("empty", 5.0, "rejected (empty answer)")):
            with self.subTest(behaviour=behaviour):
                started = time.monotonic()
                result = self.ask(behaviour, timeout)
                self.assertLess(time.monotonic() - started, 1.4)  # the timeout is honoured
                self.assertEqual((result["status"], result["method"], result["answer"]),
                                 ("answered", "extractive", extractive))
                self.assertTrue(any(expected in w for w in result["warnings"]), result["warnings"])
                self.assertTrue(result["citations"])

    def test_server_gone_falls_back_too(self):
        closed = FakeOllama()
        url = closed.url
        closed.close()
        generator = ans.ollama_generator("m", url, timeout=2)
        result = ans.answer(QUESTION, self.store, self.store.embeddings, 3, 0.1, 2, generator)
        self.assertEqual(result["method"], "extractive")
        self.assertTrue(any("generator failed (URLError" in w for w in result["warnings"]))

    def test_status_reports_problems_without_raising(self):
        self.ollama.behaviour = "good"
        self.assertTrue(ans.ollama_status(self.ollama.url, "m")["available"])
        self.assertIn("not pulled", ans.ollama_status(self.ollama.url, "other")["reason"])
        self.ollama.behaviour = "bad_tags"
        self.assertFalse(ans.ollama_status(self.ollama.url, "m")["available"])
        self.ollama.behaviour = "good"


class TestWorkflowWithOllama(unittest.TestCase):
    """The workflow wiring (`runtime.make_tools`) with the local model unavailable or failing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        write_corpus(root)
        store = build(root)[0]
        self.settings = {**SETTINGS, "rag": {**SETTINGS["rag"], "min_relevance": 0.1}}
        self.patch = unittest.mock.patch.object(pipeline, "open_store", return_value=(store, {}, store.embeddings))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_unavailable_model_is_noted_once_and_answers_stay_extractive(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "no server"}):
            tools, notes = runtime.make_tools(self.settings, use_llm=True)
            first, second = tools.rag_answer(QUESTION), tools.rag_answer(QUESTION)
        self.assertEqual(notes, ["local model unavailable (no server); extractive answers used"])
        self.assertEqual((first["method"], second["method"]), ("extractive", "extractive"))

    def test_workflow_recovers_when_the_server_starts(self):
        """Tools created while Ollama was down use it on the next document question once it answers."""
        statuses = [{"available": False, "reason": "no server"}, {"available": True, "version": "t"}]
        good = lambda *args, **kwargs: (lambda question, evidence: (GOOD["response"], {"model": "m"}))
        with unittest.mock.patch.object(ans, "ollama_status", side_effect=statuses) as status, \
                unittest.mock.patch.object(ans, "ollama_generator", good):
            tools, notes = runtime.make_tools(self.settings, use_llm=True)
            before = tools.rag_answer(QUESTION)
            self.assertEqual(notes, ["local model unavailable (no server); extractive answers used"])
            after, again = tools.rag_answer(QUESTION), tools.rag_answer(QUESTION)
        self.assertEqual((before["method"], after["method"], again["method"]), ("extractive", "generative", "generative"))
        self.assertEqual(notes, [])  # the note no longer applies
        self.assertEqual(status.call_count, 2)  # once available, the server is not checked again

    def test_market_and_distress_context_never_contacts_the_model(self):
        """The toggle applies to document questions only (D17): context answers are extractive, unchecked."""
        with unittest.mock.patch.object(ans, "ollama_status", side_effect=AssertionError("model checked")), \
                unittest.mock.patch.object(ans, "ollama_generator", side_effect=AssertionError("model used")):
            tools, notes = runtime.make_tools(self.settings, use_llm=True)
            for branch in ("market", "distress"):
                self.assertIsNone(tools.context_answer(branch).get("generation"))
        self.assertEqual(notes, [])

    def test_generator_failure_reaches_the_workflow_warnings(self):
        def failing(*args, **kwargs):
            def generate(question, evidence):
                raise TimeoutError("timed out")
            return generate
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": True, "version": "t"}), \
                unittest.mock.patch.object(ans, "ollama_generator", failing):
            tools, notes = runtime.make_tools(self.settings, use_llm=True)
            state = build_graph(tools).invoke({"question": QUESTION})
        self.assertEqual(state["trace"], ["route_request", "retrieve_context", "compose_answer"])
        self.assertTrue(state["answer"].startswith("Liquidity risk"))
        self.assertTrue(any("generator failed (TimeoutError" in w for w in state["warnings"]), state["warnings"])
        self.assertEqual(notes, [])


sys.path.insert(0, str(PROJECT_ROOT / "app"))
import ui  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

READY = ld.rag_index_status(SETTINGS["rag"])["ready"] and model_cached(MODEL)


@unittest.skipUnless(READY, "RAG index not current or embedding model not cached")
class TestDashboardWithOllama(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def ask(self, page: str, text_key: str, toggle_key: str, question: str) -> AppTest:
        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "views" / f"{page}.py"), default_timeout=180).run()
        at.toggle(key=toggle_key).set_value(True).run()
        at.text_input(key=text_key).input(question)
        return next(b for b in at.button if b.label in ("Ask", "Run workflow")).click().run()

    def test_rag_page_reports_an_unavailable_model(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "no server"}):
            at = self.ask("rag_assistant", "rag_text", "rag_llm", "Which VIX value is used for date t?")
        self.assertFalse(at.exception)
        self.assertTrue(any("local model unavailable (no server)" in w.value for w in at.warning))
        self.assertTrue(any("Method: extractive" in c.value for c in at.caption))

    def test_rag_page_reports_a_failed_generation(self):
        def generate(question, evidence):
            raise TimeoutError("timed out")
        with unittest.mock.patch.object(ui, "ollama_generator", return_value=(generate, None)):
            at = self.ask("rag_assistant", "rag_text", "rag_llm", "Which VIX value is used for date t?")
        self.assertFalse(at.exception)
        self.assertTrue(any("generator failed (TimeoutError" in w.value for w in at.warning))

    def test_workflow_page_reports_an_unavailable_model(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "no server"}):
            at = self.ask("research_workflow", "wf_text", "wf_llm", "Which VIX value is used for date t?")
        self.assertFalse(at.exception)
        self.assertTrue(any("local model unavailable (no server)" in w.value for w in at.warning))
        self.assertTrue(any("Method: extractive: the local model could not be used" in c.value for c in at.caption))

    def test_workflow_page_shows_a_rejected_generation(self):
        bad = lambda *args, **kwargs: (lambda question, evidence: ("An answer without citations.", {"model": "m"}))
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": True, "version": "t"}), \
                unittest.mock.patch.object(ans, "ollama_generator", bad):
            at = self.ask("research_workflow", "wf_text", "wf_llm", "Which VIX value is used for date t?")
        self.assertFalse(at.exception)
        self.assertTrue(any("Method: extractive fallback: m's answer was rejected by the grounding check (no citations)"
                            in c.value for c in at.caption), [c.value for c in at.caption])

    def test_workflow_page_shows_market_context_as_extractive_without_model_notes(self):
        with unittest.mock.patch.object(ans, "ollama_status", return_value={"available": False, "reason": "no server"}):
            at = self.ask("research_workflow", "wf_text", "wf_llm", "What is the current market stress level?")
        self.assertFalse(at.exception)
        self.assertTrue(any("Method: extractive document context" in c.value for c in at.caption))
        self.assertFalse(any("local model unavailable" in w.value for w in at.warning))


LIVE = os.environ.get("FINCRISIS_LIVE_OLLAMA") == "1"


@unittest.skipUnless(LIVE and READY, "set FINCRISIS_LIVE_OLLAMA=1 (and have a current index) to run against Ollama")
class TestLiveOllama(unittest.TestCase):
    """Opt-in: the configured local model, end to end. Checks structure and grounding, not wording."""

    def test_answer_with_the_local_model(self):
        rag = SETTINGS["rag"]
        o = rag["ollama"]
        status = ans.ollama_status(o["url"], o["model"])
        if not status["available"]:
            self.skipTest(status["reason"])
        store, _, embeddings = pipeline.open_store(rag)
        generator = ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"], rag["max_answer_sentences"])
        result = ans.answer("Which VIX value is used for date t?", store, embeddings, rag["top_k"], rag["min_relevance"],
                            rag["max_answer_sentences"], generator, o["min_support"], pipeline.make_retriever(store, rag))
        self.assertEqual(result["status"], "answered")
        self.assertIn(result["method"], ("generative", "extractive"))
        self.assertIsNotNone(result["generation"])  # the model was called
        ranks = {e["rank"] for e in result["evidence"]}
        self.assertTrue({c["marker"] for c in result["citations"]} <= ranks)
        if result["method"] == "generative":  # an accepted answer passed the grounding check
            evidence = {e["rank"]: e["text"] for e in result["evidence"]}
            self.assertEqual(ans.check_grounding(result["answer"], evidence, o["min_support"], result["question"]), [])
        else:
            self.assertTrue(result["warnings"])


if __name__ == "__main__":
    unittest.main()
