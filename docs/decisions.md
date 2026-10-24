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
  - **Strong class imbalance:** 220 of 6,819 rows have label 1 (3.23%).
  - **Mixed scales (updated 2026-10-24):** 71 features lie in [0, 1], but 24
    features also contain values above 1, almost all above 10⁶ (rounded to 3
    significant digits). Their cause is unknown: UCI documents no units,
    definitions, or preprocessing. They are kept unchanged and treated as
    observations requiring investigation, not as errors.
  - **Target labels unverified (updated 2026-10-24):** UCI does not document
    what 0 and 1 mean, and the original paper was not accessible. No
    assumption is made about which class represents bankruptcy.
  - **No company identifier:** the same company could appear in several rows,
    so a random split may place it in both train and test.
  - **Feature timing undocumented** relative to the bankruptcy event.

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

---

## D8. Company distress preprocessing and split

- **Date:** 2026-10-24
- **Decision:** `scripts/preprocess_taiwan_bankruptcy.py` writes
  `data/interim/taiwan_bankruptcy/` using only deterministic,
  target-independent steps:
  - Exact parsing (`float_precision="round_trip"`).
  - ASCII snake_case names (`Bankrupt?` -> `bankrupt`), with the original
    names and positions in `column_map.csv`. Collisions are rejected.
  - Four columns dropped, each re-verified on the data at run time:
    `Current Liability to Liability` (identical to `Current
    Liabilities/Liability`), `Current Liability to Equity` (identical to
    `Current Liabilities/Equity`), `Net worth/Assets` (equals
    1 - `Debt ratio %`), and `Net Income Flag` (constant).
  - Kept unchanged: `Liability-Assets Flag` (near-constant), `Operating Gross
    Margin` and `Gross Profit to Sales` (near-identical, not identical), and
    all 24 large-value features.
  - Stratified 80/20 train/test split, seed 42, saved in `split.csv`.
- **Leakage rules:** nothing is fitted in preprocessing. Every data-dependent
  step (large-value treatment, imputation, scaling, selection, class
  weights) is fitted inside a pipeline on training folds only. Model choices
  use stratified 5-fold CV within the training set; the test set is used once.
- **Metrics:** PR-AUC (primary), ROC-AUC, precision and recall at a chosen
  threshold, Brier score, calibration, and bootstrap confidence intervals.
  Accuracy is not a headline metric.

---

## D9. Company distress baselines

- **Date:** 2026-10-24
- **Decision:** `scripts/train_distress_baselines.py` compares DummyClassifier
  (prior), Logistic Regression and XGBoost with fixed settings from
  `config/settings.yaml` (no tuning).
- **Label caveat:** label 1 is treated as the class to detect only because it
  is the minority class. Its meaning (and that of 0) is unverified; reports
  must not describe results as predicting bankruptcy.
- **Two stages:**
  - `--stage cv` loads training rows only (5-fold x 3-repeat stratified CV,
    seed 42), freezes one treatment and one max-F1 threshold per model family
    in `selection.json`, and never reads test labels.
  - `--stage final` runs only after review: it fits each frozen configuration
    on the full training set and evaluates the test set once. All models are
    reported; no winner is chosen on test results.
- **Large-value treatments**, all inside pipelines and refitted per fold:
  LR with standard scaling of the original values, LR with a quantile
  transform, LR and XGBoost with values above 1 treated as missing, and
  XGBoost on the original values. No indicator columns.
- **Diagnostics only:** `lr_quantile_drop24` and `xgb_raw_drop24` remove the
  large-value columns as an ablation. They are never selected, and a score
  change does not show that those features are defective.
- **Large-value column list (resolved 2026-10-24):** defined as the columns
  with any **training-row** value above `large_value_threshold`, so no test
  rows influence it. This gives 23 columns, not the 24 found over all rows in
  Milestones 2 and 3: `current_ratio` has a single value above 1 (row_id
  2499), and that row is in the test split. The feature is kept unchanged and
  is not assumed erroneous. The list affects only the drop-24 diagnostics and
  the grouped permutation analysis; `LargeValuesToMissing` applies the
  threshold to every column, so that test value is still treated
  consistently. The `_drop24` names refer to the original inspection; reports
  state the actual count. Both lists are recorded in `selection.json`.
  The list is computed once from all training rows (feature values only, no
  labels), so within CV it also reflects validation-fold feature values; it
  is a label-free column choice and does not use test rows.
- **Selection rule (fixed before running):** per family, the first treatment
  in preference order (LR: quantile, large-as-missing, standard; XGBoost: raw,
  large-as-missing) whose mean fold-by-fold average-precision deficit to the
  best treatment is at most one standard error.
- **Class weights** come from each fit's labels: `class_weight="balanced"` and
  XGBoost `scale_pos_weight` = count(0) / count(1).
- **Artifacts:** reports in `data/reports/taiwan_bankruptcy/baselines/`,
  models in `artifacts/models/taiwan_bankruptcy/` (both ignored by Git).
