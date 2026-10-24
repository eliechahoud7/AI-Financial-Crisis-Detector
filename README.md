# AI Financial Crisis Detector

A local, offline-first system that:

1. Predicts company financial distress (XGBoost).
2. Forecasts S&P 500 market stress over the next 20 trading days (XGBoost).
3. Detects market anomalies (Isolation Forest).
4. Classifies market regimes (K-Means).
5. Generates evidence-based risk-mitigation recommendations (RAG + local LLM via Ollama).

Everything runs locally. No paid APIs, cloud services, Docker, or fine-tuning.

> **Disclaimer:** This is a research and educational project. Its outputs are
> probabilistic estimates with known limitations, not financial advice, and
> they do not guarantee any financial outcome.

## Status

Company distress model (UCI Taiwanese Bankruptcy dataset), in progress:

- Download, inspection, and preprocessing: done (`scripts/download_taiwan_bankruptcy.py`,
  `scripts/inspect_taiwan_bankruptcy.py`, `scripts/preprocess_taiwan_bankruptcy.py`).
- Baseline models (`scripts/train_distress_baselines.py`): cross-validation and the one-time
  held-out test evaluation are done. Results, limitations, and artifacts: `docs/model_card.md`.
- The meanings of the dataset's target labels are unverified (see `docs/decisions.md`, D4).

Market-stress prediction, anomaly detection, regimes, RAG, and the dashboard have not started.

## Setup

Requires Python 3.12 (Windows `py` launcher shown).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Dependency ranges are declared in `pyproject.toml`, and exact versions are in
`requirements-lock.txt`. Tests that need the downloaded dataset are skipped
if it is absent. For the full data and model pipeline, see
`docs/model_card.md` (section 12).

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
do not tune settings against it. Write a new held-out set for any further
tuning.

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
