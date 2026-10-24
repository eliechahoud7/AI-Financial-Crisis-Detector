# Design Decisions

Each entry records what was decided and why. Update an entry, with a date,
if the decision changes.

---

## D1. Python 3.12 in a project virtual environment

- **Date:** 2026-10-24
- **Decision:** Use Python 3.12.10 in `.venv`, created with `py -3.12 -m venv .venv`.
  The system Python 3.13.5 install stays unchanged.
- **Why:** The scientific and ML packages planned for this project have mature
  3.12 wheels. A project venv keeps dependencies isolated.

---

## D2. Market-stress target definition

- **Date:** 2026-10-24
- **Decision:** For each trading day *t* (the decision date, using the closing
  price `C_t` of the S&P 500, `^GSPC`):

  ```
  DD_t = min over j in {t+1, ..., t+H} of ( C_j / max(C_t, ..., C_j) - 1 )

  y_t  = 1  if DD_t <= -0.05
         0  otherwise
  ```

  with `H = 20` trading days.

  In words: `y_t = 1` if, during the next 20 trading days, the index closes at
  least 5% below its highest close since day *t* (including day *t* itself).

- **Details:**
  - **The peak is the forward running maximum.** It starts at `C_t` and includes
    any new highs inside the window. A rally followed by a 5% drop therefore
    counts.
  - **Why not a trailing peak (e.g. the 52-week high):** if the index were already
    5% below that peak at *t*, the label would be known at prediction time. The
    model would learn "we are already in a drawdown" instead of forecasting a
    new decline.
  - **Trading days** are rows in the index price history. Weekends and holidays
    are excluded automatically.
  - **Closing prices only.** An intraday breach that recovers by the close does
    not count.
  - **The last H rows have no label** (incomplete future window). They are
    dropped from training and evaluation, never filled with 0.
  - **Equality counts as an event** ("at least 5%"). A small tolerance will be
    used in code to avoid floating-point edge cases.
- **Base rate:** to be measured from the data, not assumed. Positive labels on
  consecutive days are highly correlated, because their windows overlap.

---

## D3. Chronological evaluation and leakage prevention

- **Date:** 2026-10-24
- **Decision:**
  - Features for day *t* use only data available at the close of *t*.
  - Train, validation, and test sets are split in time order, never randomly.
    Walk-forward evaluation is the primary method.
  - **Purge gap:** the label for day *t* depends on prices up to *t+20*, so the
    last 20 training rows before each test period are removed
    (`purge_gap_trading_days >= horizon_trading_days`).
  - Scalers, imputers, and any other fitted transforms are fit on training data only.
  - Each model must beat simple baselines (e.g. always predicting the training
    base rate, or a rule based on recent volatility).
  - Metrics: PR-AUC, Brier score, and calibration. Accuracy is not used as a
    headline metric because the classes are imbalanced.
- **Why:** overlapping label windows and time dependence make random splits
  produce over-optimistic results.

---

## D4. Company distress dataset: UCI Taiwanese Bankruptcy Prediction

- **Date:** 2026-10-24
- **Decision:** Use the UCI Taiwanese Bankruptcy Prediction dataset
  (CC BY 4.0) for the initial distress model.
- **Known limitations (to be confirmed when the data is loaded):**
  - **Country:** Taiwanese companies only, using Taiwan Stock Exchange
    bankruptcy criteria. The model **may not generalize** to other countries,
    accounting standards, or market structures.
  - **Period:** data from roughly 1999–2009. Results may not hold for current
    economic conditions.
  - **No time column per row:** a strict chronological split is not possible.
    This is a cross-sectional classifier, not a time-based forecast, and results
    are reported with that caveat.
  - **Strong class imbalance:** few bankrupt companies.
  - **Features are already normalized:** the original units are not available.

---

## D5. Local LLM: Qwen2.5-3B via Ollama

- **Date:** 2026-10-24
- **Decision:** Start with `qwen2.5:3b` (Q4_K_M, about 1.9 GB) through Ollama,
  using CPU inference.
- **License check (done 2026-10-24):** the Qwen Research License Agreement allows
  **non-commercial use only** ("for research or evaluation purposes only").
  Commercial use requires a separate license from Alibaba Cloud. Acceptable for
  this research and educational project.
- **Commercial fallback:** Qwen2.5-7B is licensed under Apache 2.0 but is larger and slower.
  The model name will be set in config so it can be switched.
- **Hardware:** the GPU has 2 GB of VRAM, so inference runs on CPU, with about
  16 GB of system RAM available.

---

## D6. Market data source: yfinance

- **Date:** 2026-10-24
- **Decision:** Use yfinance for historical index data.
- **Caveats:** unofficial Yahoo Finance access, intended for personal and
  non-commercial use. It can be rate-limited or break without notice. Raw
  downloads are cached in `data/raw/` with the download date recorded.

---

## D7. Incremental scope

- **Date:** 2026-10-24
- **Decision:** Build one milestone at a time. Folders and dependencies for later
  milestones (RAG, agents, UI) are added only when that milestone starts.
