# Architecture

## Overview

```
yfinance / UCI dataset
        │
        ▼
 data/raw ──► data loading + validation ──► feature engineering ──► data/processed
                                                     │
                                  ┌──────────────────┴──────────────────┐
                                  ▼                                     ▼
                         Market-stress model                    Distress model
                  (rule baseline, LogReg, XGBoost)            (LogReg, XGBoost)
                                  └──────────────────┬──────────────────┘
                                                     ▼
                                         model outputs + metrics
                                                     │
     knowledge_base/ + docs ──► FAISS index ──► LangGraph workflow ──► Ollama (local LLM)
                                                     │
                                                     ▼
                                            Streamlit dashboard
```

## Components

| Component | Location | Responsibility |
|---|---|---|
| Configuration | `config/settings.yaml` | Single source for tickers, target parameters, seeds, paths |
| Data loading | `src/fincrisis/data/` | Download, cache, and validate raw data |
| Features | `src/fincrisis/features/` | Backward-looking features only (data up to day *t*) |
| Models | `src/fincrisis/models/` | Training and prediction for the market-stress and company-distress models |
| Evaluation | `src/fincrisis/evaluation/` | Chronological splits, metrics, baselines |
| RAG | `src/fincrisis/rag/` | Document ingestion, embeddings, retrieval with citations, optional local LLM |
| Agents | `src/fincrisis/agents/` | LangGraph workflow routing questions to the market signal, distress results, or documents (D17) |
| QA | `src/fincrisis/qa/` | Read-only integrity, reproducibility, RAG regression, environment and validation-set checks; `scripts/run_qa.py` runs them all (D22–D28) |
| Scripts | `scripts/` | Reproducible command-line pipeline steps |
| Dashboard logic | `src/fincrisis/dashboard/` | Read-only, integrity-checked loaders, disclosure texts, Plotly chart builders (D18–D20) |
| App | `app/` | Streamlit entry point (`streamlit_app.py`), five pages (`views/`) and presentation helpers (`ui.py`); reads saved artifacts, never trains (D21) |

## RAG: LangChain and custom components

LangChain provides the document, splitting, embedding and vector-store layers
of the RAG. The layers that decide what an answer says are plain Python in
`src/fincrisis/rag/`.

| Part | Uses | Where |
|---|---|---|
| Documents with source and section metadata | LangChain (`langchain_core.documents.Document`) | `documents.py` |
| Chunking by Markdown heading, then whole blocks within the embedding model's token limit (D30) | Own parser; the model's tokenizer counts tokens | `chunking.py` (the original character splitter, LangChain's `MarkdownHeaderTextSplitter` and `RecursiveCharacterTextSplitter`, stays in `documents.py`) |
| Embeddings (all-MiniLM-L6-v2; revision recorded in the index manifest) | LangChain (`langchain_huggingface.HuggingFaceEmbeddings`) | `pipeline.py` |
| Vector store | LangChain `VectorStore` interface, so `as_retriever()` works; custom FAISS implementation inside | `index.py` (`FaissStore`) |
| Dense, hybrid and section-scoped retrieval | Custom Python | `retrieval.py`, `answer.py` |
| Extractive answers, `[n]` citations, abstention | Custom Python | `answer.py` |
| Ollama generation and grounding check | Custom Python (fixed prompt over the local HTTP API) | `answer.py` |
| Retrieval and generation evaluation | Custom Python | `evaluation.py`, `scripts/evaluate_rag*.py` |

Why the custom parts are kept:

- **FAISS store** (D14): `langchain-community`'s FAISS wrapper saves its
  document store with pickle, so loading an index could run code, and the
  package reports that it is being sunset. `FaissStore` saves `index.faiss`
  plus `chunks.json`. Its manifest of checksums makes loading refuse an
  altered index, a stale one, or one built with a different embedding model.
- **Retrieval and citations** (D14, D16, D17): answers must quote retrieved
  passages and cite them by number, and must abstain below the fixed
  `min_relevance`. The scores, the threshold and the dense default were
  evaluated as implemented; reimplementing them with other abstractions would
  change the scores that those results describe.
- **Grounding check and prompt** (D15, D16): a generated answer is accepted
  only if every sentence cites retrieved evidence, and its numbers and most
  of its words occur in the cited passages. The prompt and the deterministic model
  options are part of the recorded generation results, so they stay fixed.
- **Evaluation safeguards** (D16): the development and validation question
  sets, the validation-set checksum and the rule against tuning on the
  validation set rely on this code staying the same as what was evaluated.

## Principles

- **Logic in `src/` only.** Notebooks, scripts, and the app import from the package.
- **No leakage.** Features use information available at the close of day *t*;
  labels use days *t+1 … t+H*; evaluation is chronological with a purge gap.
- **Reproducible.** Seeds and parameters come from config; raw data is never edited.
- **Honest outputs.** Predictions are probabilities with reported uncertainty and
  limitations. The LLM may cite only retrieved sources and must say when evidence
  is insufficient.

## Milestones

Components, in the order they are being built:

1. **Company distress model** (UCI Taiwanese Bankruptcy dataset): done,
   including the one-time held-out test (see `docs/model_card.md`).
2. **Market data and market-stress prediction**: done, including the
   one-time held-out test (D10–D13).
3. **RAG + local LLM**: done (D14–D16).
4. **LangGraph workflow** connecting the market signal, the distress
   results, and the documents: done (D17).
5. **Streamlit dashboard**: done (D18–D21). Launch from the project root with
   `.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py`; it serves on
   `localhost` only, with usage statistics off (`.streamlit/config.toml`).

Deferred, not part of the current roadmap: anomaly detection and market
regime classification. They were planned originally but will not be built
unless the roadmap changes.

Folders for later milestones are created when their milestone starts.
