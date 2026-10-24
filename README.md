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

Milestone 1 (market data and market-stress prediction): project skeleton only.
No models are implemented yet.

## Setup

Requires Python 3.12 (Windows `py` launcher shown).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dependencies will be listed in `pyproject.toml` as each phase begins.

## Layout

| Path | Purpose |
|---|---|
| `config/settings.yaml` | Tickers, target definition, seeds, paths |
| `src/fincrisis/` | All project logic (data, features, models, evaluation) |
| `scripts/` | Command-line entry points that call `src/` |
| `notebooks/` | Exploration and reporting only |
| `app/` | Streamlit UI (later milestone) |
| `tests/` | Automated checks, including leakage guards |
| `docs/` | Architecture and design decisions |
| `data/`, `artifacts/` | Local data and trained models (not committed) |

See `docs/architecture.md` and `docs/decisions.md` for details.
