# Architecture

## Overview

```
yfinance / UCI dataset
        │
        ▼
 data/raw ──► data loading + validation ──► feature engineering ──► data/processed
                                                     │
              ┌──────────────────┬──────────────────┼──────────────────┐
              ▼                  ▼                  ▼                  ▼
     Market-stress model   Anomaly detector   Regime classifier   Distress model
        (XGBoost)         (Isolation Forest)      (K-Means)          (XGBoost)
              └──────────────────┴────────┬─────────┴──────────────────┘
                                          ▼
                              model outputs + metrics
                                          │
     knowledge_base/ ──► FAISS index ──► LangGraph workflow ──► Ollama (local LLM)
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
| Models | `src/fincrisis/models/` | Training and prediction for the four models |
| Evaluation | `src/fincrisis/evaluation/` | Chronological splits, metrics, baselines |
| RAG (later) | `src/fincrisis/rag/` | Document ingestion, embeddings, retrieval with citations |
| Agents (later) | `src/fincrisis/agents/` | LangGraph workflow that grounds recommendations in retrieved sources |
| Scripts | `scripts/` | Reproducible command-line pipeline steps |
| App (later) | `app/` | Streamlit UI; loads saved artifacts, never trains |

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

1. **Company distress model** (UCI Taiwanese Bankruptcy dataset), in progress:
   download, inspection, preprocessing, and baseline cross-validation are done;
   the held-out test evaluation is pending.
2. Market data and market-stress prediction
3. Anomaly detection and regime classification
4. RAG + local LLM recommendations
5. Streamlit dashboard

Folders for later milestones are created when their milestone starts.
