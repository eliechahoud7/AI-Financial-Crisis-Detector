"""Streamlit helpers shared by the dashboard pages (docs/decisions.md, D21).

Only presentation lives here: caching, notices and layout. Loading, integrity
checks and figures come from `fincrisis.dashboard` (loaders, charts); RAG
answers and the workflow come from `fincrisis.rag`
and `fincrisis.agents`. Nothing here writes files, downloads data or fits
models.

Cached results are keyed on a fingerprint (path, size, modification time) of
the files they read, so an artifact changed on disk is loaded, and therefore
verified, again. (Streamlit does not hash arguments named with a leading
underscore, so the fingerprint argument is called `key`.)
"""

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from fincrisis.dashboard import charts, loaders

CODE_ROOT = loaders.PROJECT_ROOT
# Where the data and artifacts are read from. Defaults to the project; tests point it elsewhere.
# The integrity manifest is always read from the code checkout (D19).
DATA_ROOT = Path(os.environ.get("FINCRISIS_DATA_ROOT", CODE_ROOT))


def settings() -> dict:
    return loaders.load_settings(CODE_ROOT)


def paths() -> dict:
    return settings()["paths"]


def fingerprint(*relative_dirs: str) -> tuple:
    """(path, size, mtime) of every file under the given directories of DATA_ROOT, plus the manifest."""
    entries = []
    for directory in relative_dirs:
        base = DATA_ROOT / directory
        if base.is_dir():
            entries += [(str(p.relative_to(DATA_ROOT)), p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(base.rglob("*")) if p.is_file()]
    manifest = loaders.FROZEN_MANIFEST
    if manifest.exists():
        entries.append(("manifest", manifest.stat().st_size, manifest.stat().st_mtime_ns))
    return tuple(entries)


def _market_dirs() -> tuple:
    p = paths()
    return (f"{p['raw_data']}/market", f"{p['processed_data']}/market_stress", f"{p['data_reports']}/market_stress",
            f"{p['models']}/market_stress")


def _distress_dirs() -> tuple:
    p = paths()
    return (f"{p['data_reports']}/taiwan_bankruptcy/baselines", f"{p['models']}/taiwan_bankruptcy")


@st.cache_data(show_spinner="Verifying the frozen market snapshot...")
def _snapshot(key: tuple, root: str) -> dict:
    return loaders.market_snapshot(Path(root), paths())


@st.cache_data(show_spinner="Computing features from the snapshot...")
def _features(key: tuple, root: str) -> pd.DataFrame:
    return loaders.market_features(loaders.market_snapshot(Path(root), paths()))[0]


@st.cache_data(show_spinner="Verifying the frozen market-stress results...")
def _stress(key: tuple, root: str) -> dict:
    return loaders.market_stress_results(Path(root), paths())


@st.cache_data(show_spinner="Scoring the latest snapshot day with the frozen models...")
def _signal(key: tuple, root: str, stale_after_days: int, day: str) -> dict:
    return loaders.current_market_signal(Path(root), paths(), stale_after_days)


@st.cache_data(show_spinner="Verifying the frozen company-distress results...")
def _distress(key: tuple, root: str) -> dict:
    return loaders.distress_results(Path(root), paths())


def _attempt(fn, *args):
    """The loader's result, or None after showing why it is unavailable and how to fix it."""
    try:
        return fn(*args)
    except loaders.ArtifactError as error:
        show_artifact_error(error)
        return None


def snapshot():
    return _attempt(_snapshot, fingerprint(f"{paths()['raw_data']}/market"), str(DATA_ROOT))


def features():
    return _attempt(_features, fingerprint(f"{paths()['raw_data']}/market"), str(DATA_ROOT))


def stress_results():
    return _attempt(_stress, fingerprint(*_market_dirs()), str(DATA_ROOT))


def current_signal():
    stale = settings()["agent"]["stale_after_days"]
    today = pd.Timestamp.now(tz="UTC").date().isoformat()  # new day, new staleness check
    return _attempt(_signal, fingerprint(*_market_dirs()), str(DATA_ROOT), stale, today)


def distress_results():
    return _attempt(_distress, fingerprint(*_distress_dirs()), str(DATA_ROOT))


def show_artifact_error(error: loaders.ArtifactError) -> None:
    st.error(f"**Unavailable:** {error}", icon=":material/error:")
    if error.remedy:
        st.markdown(f"**How to fix:** {error.remedy}")


def dark_mode() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # no browser context (tests)
        return False


# Plotly toolbar without Zoom, Pan, Autoscale and Reset axes; hover, legends and the other buttons are kept.
TOOLBAR = {"modeBarButtonsToRemove": ["zoom2d", "pan2d", "autoScale2d", "resetScale2d"]}


def figure(fig, table: pd.DataFrame | None = None, key: str | None = None, caption: str | None = None,
           config: dict | None = None) -> None:
    """Show a chart in the current theme, with an optional table view of the same numbers."""
    st.plotly_chart(charts.themed(fig, dark_mode()), theme=None, key=key, width="stretch",
                    config=TOOLBAR if config is None else config)
    if caption:
        st.caption(caption)
    if table is not None:
        with st.expander("Table view"):
            st.dataframe(table, hide_index=True, width="stretch")


def chart(builder, *args, table: pd.DataFrame | None = None, key: str | None = None, caption: str | None = None,
          config: dict | None = None, **kwargs) -> None:
    """Build a chart with `charts.build_or_message`, so missing inputs show a message instead of an error."""
    figure(charts.build_or_message(builder, *args, **kwargs), table=table, key=key, caption=caption, config=config)


LOCAL_ADDRESSES = ("localhost", "127.0.0.1", "::1")


def local_only_problems() -> list[str]:
    """Why the running server is not local-only, if it is not (empty list when it is).

    `.streamlit/config.toml` is read only from the working directory, so a launch from anywhere else would
    otherwise listen on every network interface and send Streamlit's usage statistics.
    """
    problems = []
    address = st.get_option("server.address")
    if address not in LOCAL_ADDRESSES:
        problems.append(f"the server address is {address or 'unset (all network interfaces)'}, not localhost")
    if st.get_option("browser.gatherUsageStats"):
        problems.append("Streamlit usage statistics are switched on")
    return problems


def require_local_only() -> None:
    """Stop the page, showing how to relaunch, unless the server is local-only with usage statistics off."""
    problems = local_only_problems()
    if problems:
        st.error("**The dashboard is not running in local-only mode:** " + "; ".join(problems) + ".",
                 icon=":material/lock:")
        st.markdown("Stop the server and launch it from the project root, so `.streamlit/config.toml` is used:")
        st.code(r".\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py", language="powershell")
        st.stop()


def page_header(title: str, intro: str) -> dict | None:
    """Title and introduction; returns the verified market snapshot, or None after showing why it is unavailable."""
    require_local_only()
    st.title(title)
    st.markdown(intro)
    return snapshot()


def sidebar() -> None:
    with st.sidebar:
        st.markdown("**AI Financial Crisis Detector**")
        st.caption("Local research dashboard. Read-only: it never downloads data, trains models or rebuilds indexes.")


# ----------------------------------------------------------------------------- documents and workflow

def index_status() -> dict:
    """RAG index status from its manifest (never loads the model and never rebuilds the index)."""
    return loaders.rag_index_status(settings()["rag"], DATA_ROOT)


def index_fingerprint() -> tuple:
    return fingerprint(settings()["rag"]["index_dir"])


@st.cache_resource(show_spinner="Loading the document index and the local embedding model...")
def rag_store(key: tuple, root: str) -> tuple:
    """(store, embeddings, retriever) exactly as `scripts/ask_rag.py` opens them."""
    from fincrisis.rag import pipeline
    rag = settings()["rag"]
    store, _, embeddings = pipeline.open_store(rag, Path(root))
    return store, embeddings, pipeline.make_retriever(store, rag)


def ollama_generator() -> tuple:
    """The local Ollama generator with the configured settings, or (None, reason) if it is not running."""
    from fincrisis.rag import answer as ans
    rag = settings()["rag"]
    o = rag["ollama"]
    status = ans.ollama_status(o["url"], o["model"])
    if not status["available"]:
        return None, status["reason"]
    return ans.ollama_generator(o["model"], o["url"], o["timeout_seconds"], o["options"], rag["max_answer_sentences"]), None


@st.cache_resource(show_spinner="Preparing the workflow...", ttl=600)
def workflow(use_llm: bool, key: tuple, root: str) -> tuple:
    """(compiled LangGraph workflow, notes list), wired by `fincrisis.agents.runtime` as in the CLI."""
    from fincrisis.agents.graph import build_graph
    from fincrisis.agents.runtime import make_tools
    tools, notes = make_tools(settings(), use_llm, Path(root))
    return build_graph(loaders.with_frozen_checks(tools, Path(root), paths())), notes


def show_index_problem(status: dict) -> None:
    st.error(f"**The document index cannot be used:** {status['problem']}.", icon=":material/error:")
    st.markdown(f"**How to fix:** {status['remedy']} Run it from the project root:")
    st.code(r".\.venv\Scripts\python.exe -B scripts\build_rag_index.py --force", language="powershell")
    st.caption("The dashboard never rebuilds the index itself.")


CITATION_MARKER = re.compile(r"[ \t]*\[(\d+)\]")


def hide_citations(answer: str, citations: list) -> str:
    """The answer as the pages show it: without the citation markers of `citations`.

    Only markers whose number is one of the answer's citations are removed, so
    other bracketed content (intervals such as [0.380, 0.600], an uncited [2])
    stays. The answer and its citations are unchanged in the result: the
    grounding check has already run on the cited text.
    """
    markers = {c["marker"] for c in citations}
    return CITATION_MARKER.sub(lambda m: "" if int(m.group(1)) in markers else m.group(0), answer) if markers else answer
