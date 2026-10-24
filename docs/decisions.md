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

---

## D10. Market data snapshot (Milestone 5, Phase A)

- **Date:** 2026-10-24
- **Decision:** `scripts/download_market_data.py` saves unadjusted daily bars
  for `^GSPC` and `^VIX` from Yahoo Finance (yfinance, D6) to
  `data/raw/market/` (`gspc.csv`, `vix.csv`, `metadata.json`). It is kept
  separate from the company-distress data.
- **Frozen range:** from 1990-01-01 to the last completed trading date
  present in both series, 2026-10-23. `market.end_date` in the config is set
  to that date, so a rerun requests the same range. Yahoo can revise history,
  so reruns are compared using the recorded SHA-256 checksums.
- **Validation:** errors stop the download and nothing is written:
  - empty data or missing columns;
  - duplicate or unsorted dates;
  - missing or non-positive closes;
  - a first date more than 10 days after the start, or a last date more than
    7 days before the end or after it;
  - fewer than 95% of weekdays covered.

  Warnings are recorded in `metadata.json`: gaps longer than 4 calendar days,
  missing or non-positive values in other columns, inconsistent
  open/high/low/close values, and calendar differences between the two
  series. The snapshot is never replaced without `--force`.
- **Snapshot observations (2026-10-24 download):**
  - 9,271 S&P 500 rows and 9,273 VIX rows, from 1990-01-02 to 2026-10-23.
  - No missing values, duplicate dates, or non-positive prices.
  - Three gaps in both series match known US market closures: 2001-09-11 to
    2001-09-14, 2007-01-02, and 2012-10-29 to 2012-10-30.
  - VIX has rows on 2026-05-25 and 2026-09-07, which are US market holidays
    with no S&P 500 row. Their cause is unverified.
  - `^GSPC` has zero volume on 2023-05-24 (volume will not be used).
  - VIX open, high, low and close are identical on 508 rows up to 1997-11-26,
    so only the VIX close is reliable for that period.
- **VIX lag (for Phase B):** VIX features must use information from before
  day *t*. The exact rule is D11.

---

## D11. VIX alignment rule: previous S&P 500 trading day (Milestone 5, Phase B)

- **Date:** 2026-10-24
- **Type:** feature timing and leakage prevention. It does not change the
  Phase A raw snapshots in `data/raw/market/`.
- **Rule:** for every S&P 500 trading date *t*, the VIX feature value is the
  VIX close from the immediately preceding date in the **S&P 500 trading
  calendar**, not the previous row of the VIX dataset.
- **Procedure:**
  1. Align VIX to the S&P 500 calendar: keep VIX closes only on S&P 500
     trading dates, so a VIX row on a date when the S&P 500 is closed (for
     example 2026-05-25 and 2026-09-07 in the 2026-10-24 snapshot) is
     ignored.
  2. Then shift by one S&P 500 session: the value for date *t* is the aligned
     VIX close on the S&P 500 trading date immediately before *t*.
- **Missing values:** if VIX has no close on the required previous S&P 500
  session, the feature stays missing (NaN) and the count is reported. It is
  never filled from a later observation, and never from a VIX-only row such
  as a holiday print. The first S&P 500 date in the data has no previous
  session, so its value is missing.
- **Why:** VIX closes after the S&P 500 (4:15 pm versus 4:00 pm ET), so using
  the same day's VIX would assume the prediction is made after 4:15 pm.
  Lagging by one S&P 500 session guarantees the value was known before the
  close on *t*. Tying the lag to the S&P 500 calendar keeps a VIX-only
  holiday row from shifting the lag, so the meaning of "one session earlier"
  stays consistent.
- **Testing (Phase B):** tests must cover a VIX-only date, a missing
  previous-session VIX value, and that no feature for *t* uses a VIX close
  dated *t* or later.

---

## D12. Market-stress dataset, features, and splits (Milestone 5, Phase B)

- **Date:** 2026-10-24
- **Decision:** `scripts/build_market_dataset.py` reads the frozen snapshot
  (D10), verifies its checksums before and after, and writes
  `data/processed/market_stress/dataset.csv` and `metadata.json`, plus
  `data/reports/market_stress/dataset_report.md`. No model is trained and
  nothing is fitted.
- **Target:** implemented exactly as D2
  (`src/fincrisis/features/market_target.py`), with a 1e-12 tolerance so a
  decline of exactly 5% counts. The last 20 rows have no complete future
  window and are dropped, never labelled 0. Missing or non-positive prices
  stop the build; nothing is interpolated.
- **Features** (`src/fincrisis/features/market_features.py`):
  - **Timing:** S&P 500 features use closes up to and including *t*; VIX
    features follow D11.
  - **Notation:** r = daily log return; volatility = sample standard deviation
    (ddof=1) × √252.
  - **Warm-up:** every rolling window needs a full window of values;
    otherwise the feature is NaN.
  - **Definitions** (choices not fixed by the original plan are marked
    *chosen*):
    - `ret_1`, `ret_5`, `ret_20`: ln(C_t / C_{t−k}).
    - `rv_5`: volatility of the last 5 returns.
    - `rv_ratio_20_60`: rv_20 / rv_60.
    - `downside_vol_20`: √(mean of min(r, 0)² over 20 returns) × √252
      (*chosen*: semideviation around zero).
    - `drawdown_252`: C_t / max(last 252 closes) − 1.
    - `max_drawdown_20`, `max_drawdown_60`: worst peak-to-trough decline
      within the last 20 / 60 closes.
    - `dist_ma_50`: C_t / SMA50_t − 1.
    - `ma_50_slope`: SMA50_t / SMA50_{t−5} − 1 (*chosen*: 5-session relative
      change).
    - `vix_lag1`: previous-session VIX close (D11).
    - `vix_change_5`, `vix_change_20`: log change of `vix_lag1` over 5 / 20
      S&P 500 sessions (*chosen*: log change).
    - `vix_pct_252`: share of the last 252 `vix_lag1` values, including the
      current one, that are ≤ the current value (*chosen*: inclusive empirical
      percentile, historical values only).
    - `vix_rv_ratio`: `vix_lag1` / (100 × rv_20 of the previous session)
      (*chosen*: realized volatility lagged to match the VIX timing).
  - **VIX columns:** only VIX closes are used. VIX open, high and low are not
    reliable before 1997 (D10).
- **Rows:** rows with any missing feature, from the warm-up, stay in the
  dataset with `eligible = False` and a list of their missing features. They
  are not silently dropped.
- **Splits** (`src/fincrisis/evaluation/splits.py`; config
  `market_stress.*`): walk-forward with one validation block per calendar
  year from 2000 to 2014, and expanding training windows.
  - **Purge before each validation year:** training rows end 20 sessions
    (`evaluation.purge_gap_trading_days`) before the year's first session.
  - **Purge before the test period:** every development row (training or
    validation) ends 20 sessions before the first test session, 2015-01-02.
    So the last 20 sessions of 2014 are removed from validation, and no
    development label uses test-period prices.
  - **Held-out test:** every eligible row from 2015-01-01 to the last labelled
    row.
  - **No random splitting.** Phase C reuses the same function.

---

## D13. Market-stress models and evaluation rules (Milestone 5, Phase C)

- **Date:** 2026-10-24. Recorded **before** any model was run.
- **Two stages** (`scripts/train_market_stress.py`), as in D9:
  - `--stage cv` receives development rows only (positions before the first
    test session minus the purge). It runs the 15 walk-forward folds of D12
    and freezes thresholds in `selection.json`.
  - `--stage final` runs once, after review: it fits on all development rows
    and evaluates the 2015+ test set. Every model is reported; no winner is
    chosen on test results.
- **Model inputs:** exactly the 16 columns in `market_features.FEATURES`.
  `date`, `position`, `target`, `eligible` and `missing_features` are never
  inputs. Only eligible rows are used.
- **Models** (fixed settings in `config/settings.yaml`, no tuning):
  - `dummy`: training prevalence.
  - Rule baselines `rule_vix_pct_252` and `rule_downside_vol_20`: the raw
    feature value is the score. Nothing is fitted, so they cannot leak.
  - `logistic_regression`: StandardScaler, then L2 logistic regression.
  - `xgboost`: gradient-boosted trees.
- **No class weighting.** The prevalence is moderate (about 23%), and
  unweighted models keep their probabilities interpretable for the Brier
  score and calibration. This differs from D9, where the minority class was
  3%.
- **Primary metric:** average precision on the **pooled** walk-forward
  validation predictions. All 15 years are concatenated, and each validation
  row is predicted once by the model trained on its fold.
  - **Caveat:** the pooled predictions come from 15 models trained on
    different windows, so their score levels can differ between years.
  - **Secondary:** pooled ROC-AUC, plus Brier score, log-loss and calibration
    for probability models. Rules give scores, not probabilities, so their
    Brier score and log-loss are not reported.
- **Per-year metrics:**
  - Each year's average precision and ROC-AUC are reported, but are not used
    to select anything.
  - **2005 (no label-1 rows):** average precision and ROC-AUC are **not
    defined**. They are reported as `null` and excluded from any per-year
    average. They are not reported as 0.
  - **Years with fewer than 10 label-1 rows** (2013 has 2) are reported but
    flagged as unreliable.
  - The mean of the defined per-year average precisions is a secondary
    summary.
- **Thresholds:** the max-F1 threshold on pooled validation predictions,
  frozen before the test stage. The test set is scored at that threshold
  only, and is never used to tune it.
- **Uncertainty:** labels overlap (20-session windows) and label-1 rows
  cluster in about 95 episodes, so rows are not independent.
  - **Method:** a circular block bootstrap over the time-ordered sequence of
    predictions. Block length 60 sessions (three times the horizon), 2,000
    resamples, seed 42. Resamples without both classes are skipped and
    counted.
  - **Comparisons:** paired differences use the same resamples.
  - **Sensitivity:** 20- and 120-session blocks are reported for the
    primary metric.
- **Artifacts:** reports in `data/reports/market_stress/models/`, models in
  `artifacts/models/market_stress/`. Both are ignored by Git and separate
  from the company-distress component.
- **Primary model and test-set safeguards (added 2026-10-24, after the CV
  results and before any test evaluation):**
  - **Primary model:** the non-dummy model with the highest pooled
    walk-forward average precision. It is recorded in `selection.json` with
    its ranking and paired-difference evidence. The final stage still reports
    every model, and test results cannot change the primary model.
  - **Locked selection:** once the final outputs exist, `--stage cv` refuses
    to replace `selection.json`, even with `--force`.
  - **Frozen values:** a CV re-run may not change the frozen thresholds or
    the primary model unless `--accept-changed-selection` is given.
  - **Checks before the final stage:** `--stage final` refuses unless:
    - `selection.json` is complete;
    - its features, models and thresholds match what the frozen settings
      rebuild;
    - it was produced from committed code;
    - the current code is that same commit, with no uncommitted changes.
  - **Recorded in the outputs:** the final outputs record the SHA-256 of the
    `selection.json` they used.

---

## D14. Local RAG over project documents (Phase D)

- **Date:** 2026-10-24
- **Decision:** answer questions about the project's documents with local
  retrieval and cited, grounded answers (`src/fincrisis/rag/`). There are no
  paid APIs, and nothing leaves the machine except the one-time model
  download.
- **Corpus:** the `rag.sources` patterns cover `README.md`, `docs/*.md`, the
  locally generated reports (`data/reports/**/*.md`), and any `.md` or `.txt`
  files placed in `knowledge_base/documents/`. PDFs are not supported yet.
  Markdown is split by heading first, then into chunks of up to 800
  characters with 120 characters of overlap. Each chunk keeps its source path
  and section for citations.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0, 384
  dimensions, CPU).
  - **Cache only:** it is loaded only from the local Hugging Face cache.
    Downloading it requires `--allow-download`.
  - **Recorded:** the model revision is stored in the index manifest.
  - **CPU-only torch** is installed.
- **Vector store:** `FaissStore`, a small LangChain `VectorStore`
  implementation on FAISS, with inner product over L2-normalized vectors, so
  scores are cosine similarities. `langchain-community`'s FAISS wrapper was
  not used for two reasons:
  - the package reports that it is being sunset;
  - it saves its document store with pickle.

  Instead, the index is saved as `index.faiss` plus `chunks.json`, so no
  pickle is involved. A manifest stores checksums of both files and of every
  source document. Loading refuses an altered index, an index built with a
  different model, and a stale index (sources changed, removed, or added).
  Rebuilding is atomic and needs `--force`.
- **Answers (default extractive):**
  - The answer is up to 3 sentences copied from retrieved chunks, chosen by
    similarity to the question, each followed by a citation [n] to the
    retrieved chunk.
  - If the best retrieval score is below `min_relevance` (0.30 cosine), the
    reply is "insufficient evidence". The threshold was fixed before the
    evaluation and not tuned on it.
- **Optional generation:** `--llm ollama` sends the same evidence to a local
  Ollama model (D5) with instructions to cite every claim. The answer is used
  only if every citation refers to retrieved evidence; otherwise the
  extractive answer is returned with a warning. Superseded in Phase E by
  D15, which installs Ollama, adds stricter grounding checks, and evaluates
  generation.
- **Evaluation:** `knowledge_base/eval/retrieval_questions.json` has 26
  answerable questions, paraphrased and tied to an expected source and
  phrase, plus 6 unanswerable questions. Metrics are hit@1/3/5, MRR, and the
  answer and abstention rates at the threshold. A TF-IDF retriever over the
  same chunks is the baseline.
- **Separation:** artifacts are kept in `artifacts/rag/` and reports in
  `data/reports/rag/`, both ignored by Git. The market-stress and
  company-distress components are unchanged.

---

## D15. Local LLM answers through Ollama (Phase E)

- **Date:** 2026-10-24
- **Runtime:** Ollama 0.40.10, installed per user with winget (MIT
  license). It listens on `127.0.0.1:11434` only and starts with Windows.
- **Model:** `qwen2.5:3b` (Q4_K_M, about 1.9 GB, ID `357c53fb659c`), as
  chosen in D5.
  - **License:** the Qwen Research License, verified on the Ollama and
    Hugging Face pages on 2026-10-24. Non-commercial use only; distributing
    derived models would require a "Built with Qwen" notice.
  - **Hardware:** about 6.5 output tokens/s on this machine (i7-1165G7,
    16 GB RAM, partly offloaded to the 2 GB MX450).
- **Use:** generation is optional (`ask_rag.py --llm ollama`). Extractive
  answers remain the default. Settings are in `rag.ollama`, with
  deterministic decoding (temperature 0, seed 42, `num_ctx` 4096,
  `num_predict` 256).
- **Safeguards:**
  - **Before generating:** the server and the model are checked; if either
    is unavailable, the extractive answer is returned with a warning.
  - **Below the retrieval threshold:** the model is never called.
  - **Model refusal:** if the model replies that the evidence is
    insufficient, the result is "insufficient evidence".
  - **Grounding check (`check_grounding`):** a generated answer is accepted
    only if every sentence cites a retrieved passage, every number appears
    in the cited passages, and at least `min_support` (0.6) of its content
    words do. Otherwise the extractive answer is returned with a warning.
    This is a lexical check, not proof of correctness.
- **Prompt:** one revision was made before the formal evaluation and not
  tuned afterwards. It added a citation-format example on a generic topic
  outside the evaluation set, and moved the rules after the evidence.
- **Evaluation:** `scripts/evaluate_rag_generation.py` runs the 32-question
  set and reports accepted, rejected and refused shares, whether accepted
  answers cite an expected relevant passage, and latency. It lists accepted
  answers for manual review.
