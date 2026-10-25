"""Dashboard behaviour across reruns and conditions, plus an opt-in live-server smoke test (docs/decisions.md, D27).

AppTest runs the pages in this process. The live-server test starts the real
Streamlit server from the project root only when FINCRISIS_LIVE_SERVER=1 is
set, and checks that it listens on loopback addresses only and answers.
"""

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

from fincrisis.dashboard import disclosures
from fincrisis.dashboard import loaders as ld
from fincrisis.rag import pipeline
from tests.test_dashboard import DIRS, PRESENT, PROJECT_ROOT
from tests.test_rag import MODEL, model_cached

APP = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP))
import ui  # noqa: E402

SETTINGS = ld.load_settings()


def page(name: str) -> AppTest:
    return AppTest.from_file(str(APP / "views" / f"{name}.py"), default_timeout=180)


class AppCase(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()
        self.addCleanup(setattr, ui, "DATA_ROOT", ui.DATA_ROOT)


@unittest.skipUnless(PRESENT, "frozen market/distress outputs not present")
class TestReruns(AppCase):
    def test_artifact_altered_after_first_load_is_refused_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            for d in DIRS.values():
                shutil.copytree(PROJECT_ROOT / d, Path(tmp) / d)
            ui.DATA_ROOT = Path(tmp)
            at = page("market_stress").run()
            self.assertFalse(at.error, [e.value for e in at.error])
            path = Path(tmp) / DIRS["ms_reports"] / "test_predictions.csv"
            path.write_bytes(path.read_bytes().replace(b"2015-01-02", b"2015-01-01", 1))  # after the first load
            at.run()
            self.assertFalse(at.exception)
            self.assertTrue(any("test_predictions.csv does not match" in e.value for e in at.error))

    def test_stale_snapshot_warning_on_every_page(self):
        stale = SETTINGS["agent"]["stale_after_days"] + 23
        with unittest.mock.patch.object(disclosures, "snapshot_age_days", return_value=stale):
            for name in ("market_dashboard", "company_distress", "research_workflow"):
                with self.subTest(page=name):
                    at = page(name).run()
                    self.assertTrue(any(f"{stale} days old" in w.value for w in at.warning), name)


READY = ld.rag_index_status(SETTINGS["rag"])["ready"]


@unittest.skipUnless(READY, "RAG index not current")
class TestMissingEmbeddingModel(AppCase):
    def test_rag_page_explains_a_missing_embedding_model(self):
        error = RuntimeError("Embedding model x is not in the local cache. Download it once with "
                             "scripts/build_rag_index.py --allow-download.")
        at = page("rag_assistant").run()
        at.text_input(key="rag_text").input("Which VIX value is used for date t?")
        with unittest.mock.patch.object(pipeline, "open_store", side_effect=error):
            at = next(b for b in at.button if b.label == "Ask").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("could not be opened" in e.value and "--allow-download" in e.value for e in at.error))


LIVE = os.environ.get("FINCRISIS_LIVE_SERVER") == "1"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@unittest.skipUnless(LIVE and PRESENT, "set FINCRISIS_LIVE_SERVER=1 to start the real dashboard server")
class TestLiveServer(unittest.TestCase):
    def test_server_listens_on_loopback_only_and_answers(self):
        import psutil
        port = free_port()
        process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(Path("app") / "streamlit_app.py"),
                                    "--server.port", str(port)], cwd=PROJECT_ROOT,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            health = None
            for _ in range(60):
                try:
                    health = urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2).read()
                    break
                except OSError:
                    time.sleep(1)
            self.assertEqual(health, b"ok")
            listening = {c.laddr.ip for p in [psutil.Process(process.pid), *psutil.Process(process.pid).children(True)]
                         for c in p.net_connections(kind="inet") if c.status == psutil.CONN_LISTEN}
            self.assertTrue(listening)
            self.assertTrue(listening <= {"127.0.0.1", "::1"}, listening)
            page_html = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
            self.assertIn(b"<html", page_html.lower())
        finally:  # on Windows the venv's python.exe starts the real interpreter as a child: stop the whole tree
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait(timeout=30)
        self.assertFalse(psutil.pid_exists(process.pid) and psutil.Process(process.pid).status() != psutil.STATUS_ZOMBIE)


if __name__ == "__main__":
    unittest.main()
