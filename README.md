# AI Financial Crisis Detector

A local, offline-first system that:

1. Predicts company financial distress (XGBoost).
2. Forecasts S&P 500 market stress over the next 20 trading days (XGBoost).
3. Answers questions about the project with cited, evidence-based answers (RAG + local LLM via Ollama).
4. Connects these in a LangGraph workflow and a local Streamlit dashboard.

Anomaly detection and market-regime classification were planned originally and are deferred; they are
not part of the current roadmap.

Everything runs locally. No paid APIs, cloud services, Docker, or fine-tuning.

> **Disclaimer:** This is a research and educational project. Its outputs are
> probabilistic estimates with known limitations, not financial advice, and
> they do not guarantee any financial outcome.

## Status

- **Company distress** (UCI Taiwanese Bankruptcy dataset): download, inspection, preprocessing,
  baselines and the one-time held-out test are done (`docs/model_card.md`). The meanings of the
  target labels are unverified (`docs/decisions.md`, D4).
- **Market stress** (S&P 500 and VIX, 1990–2026): frozen snapshot, dataset, walk-forward
  validation and the one-time held-out test are done (D10–D13). The primary model is a
  downside-volatility rule; no model clearly beat it.
- **Document RAG**: local retrieval with cited extractive answers, optional local LLM
  generation with grounding checks (D14–D16).
- **Workflow**: a LangGraph workflow routes questions to the market signal, the distress
  results, or the documents (D17).
- **Dashboard**: a local Streamlit + Plotly dashboard over all of the above (D18–D21, D29).

## Dashboard

Five pages: Market Dashboard, Market Stress Analysis, Company Distress, Financial RAG Assistant
and AI Research Workflow. The dashboard only reads the frozen outputs: it never downloads data,
trains or refits models, regenerates predictions, or rebuilds the document index. Missing or
altered artifacts are reported with instructions instead of being shown. The Market Dashboard
explains its four indicators in plain language, shades historical market events on its charts and
ends with a summary generated from the snapshot.

The pages do not show the snapshot's as-of date, a warning when it is old, the research-only
disclaimer or the limitations of each component (D29). Those are documented in
`docs/model_card.md` and `docs/decisions.md`, and the workflow CLI still states its disclaimer
and warns about an old snapshot.

```powershell
# One-time: install the dashboard, RAG and workflow dependencies (CPU-only PyTorch)
.\.venv\Scripts\python.exe -m pip install -e ".[rag,agent,app]" --extra-index-url https://download.pytorch.org/whl/cpu
# One-time: build the document index (needed by the RAG and workflow pages)
.\.venv\Scripts\python.exe -B scripts\build_rag_index.py --allow-download

# Launch from the project root, so .streamlit/config.toml is used
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
```

Then open http://localhost:8501. `.streamlit/config.toml` binds the server to `localhost` only and
turns off Streamlit's usage statistics. The pages need the frozen outputs under `data/` and
`artifacts/` (not in Git; see `docs/model_card.md`, section 12, and D10–D13). The optional
"local model" toggles need Ollama running with `qwen2.5:3b` (non-commercial licence). After editing
any indexed document (`README.md`, `docs/`, reports), rebuild the index with
`scripts\build_rag_index.py --force`; the dashboard shows when it is out of date.

## Asking the workflow

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[rag,agent]" --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -B scripts\run_crisis_agent.py "What is the current market stress level?"
.\.venv\Scripts\python.exe -B scripts\run_crisis_agent.py "How well do the bankruptcy models perform?"
.\.venv\Scripts\python.exe -B scripts\run_crisis_agent.py "Which VIX value is used for date t?"
```

The documents must be indexed first (`scripts\build_rag_index.py`, see the RAG section below).
The market answer reads the local snapshot, so refresh it for a current signal; the answer warns
when the snapshot is more than 7 days old.

Requests for investment or trading advice (such as "Should I sell my stocks now?") are declined.
Market and distress answers come from the frozen evaluated models and are explained with cited
project documents; every answer states that it is research output, not a forecast or financial
advice. Add `--json` to see the route taken and `--llm ollama` to let the local model write
document answers.

## Setup

Requires Python 3.12 (Windows `py` launcher shown).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Set `FINCRISIS_LIVE_OLLAMA=1` to include the test against the local Ollama model
(it needs Ollama running with `qwen2.5:3b`; other Ollama tests use a stand-in server), and
`FINCRISIS_LIVE_SERVER=1` to start the real dashboard server once and check it is local-only.
Dependency ranges are declared in `pyproject.toml`, and exact versions are in
`requirements-lock.txt`. Tests that need the downloaded dataset are skipped
if it is absent. For the full data and model pipeline, see
`docs/model_card.md` (section 12).

## Quality checks

One command runs every read-only check and the test suite, with a pass/fail line per stage
(D22–D28). Run it from the project root:

```powershell
.\.venv\Scripts\python.exe -B scripts\run_qa.py            # everything (about 5 minutes on CPU)
.\.venv\Scripts\python.exe -B scripts\run_qa.py --quick    # the checks only, without the test suite
.\.venv\Scripts\python.exe -B scripts\run_qa.py --live     # also the live Ollama and dashboard-server tests
```

| Stage | Checks | Script |
|---|---|---|
| environment | installed packages match `requirements-lock.txt` | `run_qa.py` |
| integrity | frozen data, results, models and the RAG validation set match recorded checksums | `verify_integrity.py` |
| reproducibility | today's code reproduces the frozen features, target, model scores and latest signal | `check_reproducibility.py` |
| rag regression | development-set retrieval does not fall below its baseline; reference answers are unchanged | `check_rag_regression.py` |
| tests | the unit-test suite | `python -m unittest discover -s tests` |

Nothing is trained, downloaded or rebuilt. After editing an indexed document, rebuild the index,
run `scripts\check_rag_regression.py`, review any change it reports, and only then re-record the
baselines with `--update`.

## Asking questions about the project documents (RAG)

Retrieval and answering run locally on CPU (see `docs/decisions.md`, D14).
Answers quote the retrieved passages and cite them; if nothing relevant is
found, the reply says the evidence is insufficient.

```powershell
# One-time: install the RAG dependencies (CPU-only PyTorch)
.\.venv\Scripts\python.exe -m pip install -e ".[rag]" --extra-index-url https://download.pytorch.org/whl/cpu

# Build the index (the first run downloads the embedding model once)
.\.venv\Scripts\python.exe -B scripts\build_rag_index.py --allow-download

# Ask a question, and evaluate retrieval quality
.\.venv\Scripts\python.exe -B scripts\ask_rag.py "Which VIX value is used as a feature for day t?"
.\.venv\Scripts\python.exe -B scripts\evaluate_rag.py
```

Optional: let a local model write the answer (see `docs/decisions.md`, D15).
This needs [Ollama](https://ollama.com) and the `qwen2.5:3b` model, which is
licensed for non-commercial use only. Generated answers are used only if
they pass the grounding checks; otherwise the cited extractive answer is
shown.

```powershell
ollama pull qwen2.5:3b
.\.venv\Scripts\python.exe -B scripts\ask_rag.py "Which VIX value is used as a feature for day t?" --llm ollama
.\.venv\Scripts\python.exe -B scripts\evaluate_rag_generation.py
```

Retrieval uses semantic search (`rag.retrieval.mode: dense`). Hybrid
(semantic plus TF-IDF) and lexical modes exist but are not the default: they
improved the development questions but not the independent validation set
(`docs/decisions.md`, D16).

The evaluation scripts use the development questions
(`knowledge_base/eval/retrieval_questions.json`) by default. `--questions`
selects another set; the validation set
(`knowledge_base/eval/validation_questions.json`) has already been used, so
do not tune settings against it. The scripts refuse it (including renamed
copies) unless `--final-validation` is given, and then only if it still matches
its recorded checksum (D24). Write a new held-out set for any further tuning.

Add your own `.md` or `.txt` documents under `knowledge_base/documents/`,
then rebuild with `--force`. The index refuses to load if any source has
changed since it was built.

## Layout

| Path | Purpose |
|---|---|
| `config/settings.yaml` | Tickers, target definition, seeds, paths |
| `src/fincrisis/` | All project logic (data, features, models, evaluation) |
| `scripts/` | Command-line entry points that call `src/` |
| `notebooks/` | Exploration and reporting only |
| `app/` | Streamlit UI (later milestone) |
| `tests/` | Automated checks, including leakage guards |
| `docs/` | Architecture, design decisions, and the model card |
| `data/`, `artifacts/` | Local data and trained models (not committed) |

See `docs/architecture.md` and `docs/decisions.md` for details.
