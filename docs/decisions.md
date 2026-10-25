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
  and section for citations. **Updated 2026-10-25:** replaced by
  structure-aware chunking within the embedding model's token limit (D30).
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

---

## D16. RAG improvements after the Phase E evaluation

- **Date:** 2026-10-24
- **Evaluation discipline:**
  - **Development set:** `retrieval_questions.json` (32 questions) had
    already been analysed, so it is now the development set and was used to
    diagnose problems and choose settings.
  - **Validation set:** `validation_questions.json` (25 questions: 20
    answerable, 5 unanswerable) was written **before** any
    change, from facts the development set does not cover. Its SHA-256
    (`7ed9c16f…`) was recorded at creation. It is evaluated only twice: on
    the baseline (commit `e21a291`) and on the final configuration. It is
    never used to choose settings.
  - **Separate reports:** each evaluation script writes the development and
    validation results to separate reports (the `--questions` option).
- **Diagnosis (development set):**
  - 5 of the 9 unnecessary model refusals, and both wrong answers that
    passed the grounding check, had no relevant passage among the 4 passages
    sent to the model. TF-IDF ranked the answer first for 3 of those
    refusals.
  - Retrieval, not the model, was the main problem.
- **Changes:**
  1. **Hybrid retrieval** (`src/fincrisis/rag/retrieval.py`): reciprocal
     rank fusion of the dense ranking and a TF-IDF ranking (chunk text plus
     section), with `rrf_k = 60`. TF-IDF is fitted locally when the index
     loads.
  2. **Section headings embedded** with the chunk text
     (`rag.retrieval.embed_section_headings`). This is recorded in the
     manifest, and an index built with the other setting is refused.
  3. **Abstention gate:** the question is answered only if the best dense
     cosine over all chunks reaches `min_relevance` (0.30, unchanged).
     Evidence is the top-k of the configured ranking. This keeps the
     threshold's meaning while allowing lexically found passages as
     evidence.
  4. **Hedged non-answers** ("X is not directly stated", "the documents do
     not mention…") count as "insufficient evidence" instead of being
     accepted.
  5. **Grounding check:**
     - Words restated from the question, including simple word forms such
       as "looks" for "look", are not counted as unsupported claims.
     - The word-share test applies only when at least 3 other content words
       remain.
     - The number check is unchanged and applies to every sentence.
     - Answers without citations are still rejected; citations are never
       added after the fact.
- **Not changed:** the prompt, the model, `min_support` (0.6), `top_k` (4)
  and the relevance threshold. The market-stress and company-distress
  components are untouched.
- **Results.** The development set chose the changes; the validation set was
  run once on the baseline and once on the full set of changes. "Wrong
  accepted" is from a manual review of every accepted generated answer.

  | Measure | Dev baseline | Dev changes | Val baseline | Val changes |
  |---|---|---|---|---|
  | Retrieval hit@3 | 0.538 | 0.769 | 0.550 | 0.500 |
  | Retrieval hit@5 | 0.615 | 0.808 | 0.600 | 0.600 |
  | Retrieval MRR | 0.362 | 0.482 | 0.404 | 0.400 |
  | Answerable: accepted / rejected / refused | 12 / 5 / 9 | 12 / 7 / 7 | 8 / 6 / 6 | 9 / 4 / 7 |
  | Accepted but wrong | 2 | 0 | 0 | 2 |
  | Unanswerable refused | 6/6 | 6/6 | 5/5 | 5/5 |

  The retrieval gains did not hold on the validation set:
  - the hybrid's hit@3 fell from 0.550 to 0.500, with no gain at hit@5;
  - embedding the headings lowered dense hit@3 from 0.550 to 0.400;
  - two wrong answers passed the grounding check (v06 named the kept
    columns as the removed ones; v10 gave D9's class-weight setup as the
    reason the market-stress models use none).

  One question moves these rates by 0.05, so the differences are within
  noise. The evidence does not support any improvement.
- **Adopted configuration (2026-10-24, after validation):**
  - **Retrieval reverted** to the original dense search without headings
    (`rag.retrieval.mode: dense`, `embed_section_headings: false`). Dense
    mode uses the original answering path unchanged. The hybrid and lexical
    modes and the heading option remain available, with tests, but are not
    the default.
  - **Kept:** the hedged non-answer refusals (change 4), the grounding-check
    change (change 5), the evaluation tooling (`--questions`, per-mode
    retrieval comparison), and their tests.
  - **Not independently validated:** this adopted combination (original
    retrieval plus changes 4 and 5) was not measured on the validation set.
    The validation set has now been used and must not be used to tune
    settings. Changes 4 and 5 are kept on their reasoning: a hedged
    non-answer is not an answer, and words restated from the question are
    not claims. Their effect has not been isolated.
- **Next steps:** any further tuning needs a new held-out question set. A
  semantic support check could catch inverted or misattributed claims (v06,
  v10) that a lexical check cannot.

---

## D17. LangGraph workflow over market analysis and document retrieval (Phase F)

- **Date:** 2026-10-24
- **Decision:** `src/fincrisis/agents/graph.py` builds a LangGraph
  `StateGraph` that routes a question to the existing components and grounds
  the explanation in the project documents. `scripts/run_crisis_agent.py`
  runs it from the command line. LangGraph 1.2.14 is installed in `.venv`;
  `websockets` went from 17.3 to 16.1.1, which `langgraph-sdk` requires and
  `yfinance` (>= 13) accepts.
- **Graph:**
  - `route_request` sends the question to one of four branches:
    - `advice` → `refuse_advice`;
    - `market` → `assess_market`, then `report_error` if a check fails, or
      `retrieve_context` if it passes;
    - `distress` → `summarize_distress` → `retrieve_context`;
    - `documents` → `retrieve_context`.
  - On the documents branch, `retrieve_context` goes to `report_error` if
    retrieval fails, to `abstain` if no passage reaches `min_relevance`, and
    otherwise to `compose_answer`. On the market and distress branches it
    always goes to `compose_answer`: the figures are reported, followed by the
    cited context, "No supporting passage was found", or "Document context is
    unavailable" (with a warning) if retrieval fails.
  - The trace of visited nodes is returned with every answer.
- **Routing is deterministic** (keyword rules in `graph.route`), not
  model-driven, so it is testable and cannot be talked into a different
  branch:
  - investment or trading advice ("should I buy…", "do you recommend
    investing…") is declined without calling any tool;
  - market words together with "current/now/today/signal/level" go to the
    market assessment;
  - distress words together with "results/perform/accuracy" go to the
    distress results;
  - everything else goes to the document RAG.
- **Tools** (`src/fincrisis/agents/tools.py`) reuse the existing code and
  frozen outputs. They are read-only and refit or re-select nothing.
  - **`assess_market`:**
    - checks that the raw snapshot matches its checksums, and that
      `selection.json` is the one the frozen test evaluation used
      (`test_metrics.json` records its checksum);
    - computes features for the latest snapshot date with
      `market_features.build_features`;
    - scores that date with the **frozen primary model** at its frozen
      threshold, and with the saved XGBoost model, loaded only after its
      checksum matches;
    - reports the historical percentile of the primary score, the measured
      test reliability from `test_metrics.json`, and a warning if the
      snapshot is older than `agent.stale_after_days` (7).
  - **`distress_summary`:** reports the frozen held-out results of the
    company-distress baselines and the label caveat. Companies are not
    scored, because a question cannot supply the 91 required features.
  - **Documents:** the existing RAG `answer()`, with the default dense
    retrieval, extractive answers, and optional Ollama generation with
    grounding checks.
  - **Market and distress context** (`tools.context_answer`): each branch
    has a configured list of entries (`agent.context`), each a question plus
    the document and section to search. For each entry, `answer()` retrieves
    only chunks in that section (`scope`, dense cosine with the same
    `min_relevance`) and returns its best sentence. Entries without evidence
    are skipped, and citations are renumbered. The explanation is therefore
    deterministic and cites the target definition (D2), the primary-model
    rule (D13) and the test limitations for the market branch, and the test
    comparison, calibration and scope caveats for the distress branch, rather
    than whatever the user's wording retrieves. The entry questions were
    worded by checking which sentence each one selects in its section; the
    RAG validation set was not used.
- **Failures:** nothing is ever replaced by an estimate.
  - A missing or altered raw snapshot, `selection.json` or model file stops
    the market branch with an error message (`report_error`). Model files are
    checked before `joblib.load`. A failure of the secondary XGBoost model
    also stops the branch, so a partial assessment is never reported.
  - Altered or missing distress outputs are reported as "unavailable" in the
    answer, with a warning. Every answer ends with a disclaimer that this is research output,
  not a forecast or financial advice.
- **Extractive answer change** (found while testing the workflow):
  - headings, fenced code blocks and list lead-ins ending in ":" are no
    longer used as evidence sentences;
  - a sentence cut by a chunk boundary is skipped: the chunk's last sentence
    if it stops mid-sentence, and its first sentence (or the first after a
    code block) if it starts lowercase. Other sentences of the same paragraph
    are kept. If a chunk boundary cuts a code block, a heuristic decides
    whether the chunk starts or ends inside it (first fence in the first half
    of the chunk's lines means it starts inside);
  - wrapped `>` blockquote lines are rejoined into one paragraph.

  Retrieval is unchanged.
- **Not evaluated:** no question set measures the workflow's answers. Its
  tests check routing, every branch, error handling (including runs of the
  whole workflow on altered copies of the frozen outputs), abstention, and
  that the tools leave files unchanged. Answer quality is that of the components it reuses
  (D9, D13 to D16).

---

## D18. Dashboard foundation: read-only loaders and disclosures (Phase G)

- **Date:** 2026-10-24
- **Dependencies:** `streamlit` 1.66.1 and `starlette` 1.7.0 in a new `app`
  extra. Only nine packages were added; no installed package changed.
  `starlette` is held below its newest release, and 1.66.1 is the first
  Streamlit release that accepts the locked `pillow` 13.
- **Decision:** before any page is built, `src/fincrisis/dashboard/` provides
  the shared, read-only layer the pages will use:
  - **`loaders.py`** returns the frozen market snapshot, its features, the
    labelled dataset, the market-stress and company-distress results with
    their saved predictions, the current market signal, and the RAG index
    status. Nothing is written, refitted or downloaded. Checks:
    - the raw snapshot and the dataset against their recorded checksums
      (D10, D12);
    - `selection.json` against the checksum that the test evaluation and the
      saved models record; selection, test results and models must refer to
      the same dataset; saved model files against `model_metadata.json`
      (D13), without loading them;
    - the prediction files, which have no recorded checksum, by recomputing
      each model's recorded average precision and the row and label counts;
    - the RAG index against its manifest (files, embedding model, heading
      setting, stale sources), without loading the embedding model.

    A missing or inconsistent artifact raises `ArtifactError` with a remedy.
    The remedies never suggest re-running the frozen evaluations, which would
    create new results rather than restore the frozen ones.
  - **`disclosures.py`** builds the notices: the snapshot's as-of date and a
    warning once it is older than `agent.stale_after_days` (the same rule as
    the workflow), the research-only disclaimer (the workflow's own text,
    D17), and the limitations of each component. Numbers in these texts are
    read from the frozen results, not written into the code.
    **Updated 2026-10-25:** the dashboard no longer shows these notices
    (D29); `disclosures.py` still builds them, and also builds the Market
    Dashboard summary.
- **Refactor:** the workflow's tool wiring (`make_tools`) moved unchanged from
  `scripts/run_crisis_agent.py` to `src/fincrisis/agents/runtime.py`, so the
  CLI and the dashboard share it. `agents/tools.py` now exposes its checksum
  helpers (`sha256_file`, `read_json`, `verify_raw_snapshot`, `read_close`,
  `load_verified_model`) for reuse. The CLI's output was compared before and
  after the move for five questions, in text and JSON, and is identical.
- **Not done here:** no pages and no charts.

---

## D19. Checksums for frozen outputs the pipelines did not record (Phase G)

- **Date:** 2026-10-24
- **Problem:** two gaps found while reviewing the dashboard loaders (D18).
  1. The prediction files, metrics, reports and model metadata of the frozen
     evaluations have no recorded checksum. The loaders recomputed the
     recorded average precision and label counts, but an edit that keeps
     those (a changed date or row id, a rescaled score) went unnoticed, and
     `test_metrics.json` itself, which anchors `selection.json`, was not
     checked by anything.
  2. `assess_market` read `dataset.csv` for the historical percentile
     without checking it.
- **Fix for 2 (no new checksum):** `assess_market` now refuses a
  `dataset.csv` that differs from `selection.json`'s `dataset_sha256`.
  `selection.json` is itself checked against `test_metrics.json` before
  that, so the value comes from the frozen evaluation. A missing dataset
  still only omits the percentile, as before.
- **Fix for 1:** `config/frozen_artifacts.json`, a version-controlled
  manifest of 26 SHA-256 checksums (the market-stress and company-distress
  reports, predictions and saved models, and the market-stress dataset).
  The dashboard loaders check every listed file in the directories they read
  before using it; the existing checks stay as a second layer.
  - **Provenance:** the values were copied from hash records taken during
    the original runs, not computed from the current files:

    | Stage | Outputs written | Hashes recorded |
    |---|---|---|
    | Market-stress final stage | 17:30:03 | 17:30:25 |
    | Market-stress dataset | 16:40:57 | 16:41:25 |
    | Company-distress final stage | 16:05:23 | 16:27:05 |

    (local time, 2026-10-24). Before the manifest was written, every current
    file was compared with its record, and the manifest records these times.
    The current files also reproduce every recorded average precision and
    label count, and the pipelines' own checksums of `selection.json` and the
    model files.
  - **Trust:** the manifest is read from the code checkout, never from the
    data directory being checked, so a copied or altered data folder cannot
    vouch for itself. Its values must never be regenerated from the current
    files; a legitimate new evaluation would need a new, reviewed entry.
  - **Limit:** the records were taken minutes after each stage ran, within
    the same session, so they cannot prove that nothing changed in those
    minutes. The recomputed metrics are consistent with the frozen results.
- **CLI:** unchanged except that an altered `dataset.csv` now stops the
  market branch with an error instead of producing a percentile from it. The
  CLI does not read the manifest.

---

## D20. Dashboard charts (Phase G)

- **Date:** 2026-10-24
- **Decision:** `src/fincrisis/dashboard/charts.py` builds Plotly figures
  from what the dashboard loaders (D18, D19) return. It does not import
  Streamlit, read files or fit anything; every number shown comes from the
  frozen outputs passed in.
- **Charts:**
  - market history: S&P 500 and VIX closes; drawdown from the 252-session
    high and 20-day downside volatility (from the snapshot's features);
  - market-stress scores: one model's walk-forward validation and held-out
    test scores, its frozen threshold, the days at or above it ("signal"),
    observed label-1 periods, and optionally the latest snapshot day;
  - market-stress performance: average precision per model with
    block-bootstrap intervals against the base rate (test or validation),
    per-year average precision with unreliable years hollow, calibration for
    the probability-shaped models, precision-recall curves, and the confusion
    matrix at the frozen threshold;
  - company distress: average precision and ROC-AUC with bootstrap
    intervals, confusion matrices at the frozen max-F1 threshold or 0.5,
    precision-recall curves, cross-validation results per configuration, and
    the grouped permutation importance of the 23 large-value columns.
- **Conventions:** observations are neutral ink or gray shading; scores are
  colored lines with a fixed color per model, on an axis labeled as a score
  that is not a probability; the threshold is dashed and signals are labeled,
  not shown by color alone; quantities of different scale get separate
  panels, never a second y-axis. Precision-recall operating points and
  average precision in legends are the recorded values; the curves are traced
  from the saved predictions. The colors are the validated reference palette
  (slots 1-4); two are below 3:1 contrast on the light surface, so charts
  keep legends and hover text, and pages will offer table views.
- **Not possible with the saved data:**
  - per-feature importance: the pipeline saved only one grouped permutation
    result (all 23 large-value columns shuffled together), and nothing for
    the market-stress models;
  - market-stress confusion matrices at 0.5 and calibration for the rule
    models: their scores are not probabilities, and none were recorded;
  - company-distress calibration tables: not computed (model card,
    section 9);
  - company-level scores outside the test set: not available by design (D18).

---

## D21. Streamlit dashboard application (Phase G)

- **Date:** 2026-10-25
- **Decision:** `app/streamlit_app.py` is the entry point, with five pages in
  `app/views/` (not `pages/`, which Streamlit would also list on its own) and
  presentation helpers in `app/ui.py`. All loading, checks, texts and charts
  come from `fincrisis.dashboard` (D18-D20); answers come from the existing
  RAG `answer()` and the LangGraph workflow wired by `agents.runtime` (D17).
  - **Market Dashboard:** snapshot closes, drawdown and downside volatility
    (updated 2026-10-25: with indicator explanations, event annotations and
    a generated summary, D29).
  - **Market Stress Analysis:** the latest day's frozen scores, scores over
    time, and the validation and test evaluation.
  - **Company Distress:** frozen test results, cross-validation and grouped
    importance. There is no form for scoring companies.
  - **Financial RAG Assistant:** the same `answer()` call as
    `scripts/ask_rag.py`, with citations, warnings and the retrieved
    passages. If the index is missing, altered or out of date, the page says
    so, shows the rebuild command and disables questions; it never rebuilds.
  - **AI Research Workflow:** the same graph and tools as
    `scripts/run_crisis_agent.py`, so advice is refused and missing evidence
    is reported in the same way.
- **On every page:** the snapshot's as-of date (a warning once it is older
  than `agent.stale_after_days`), the research-only disclaimer, and the
  limitations from `disclosures.py`. A missing or altered artifact shows the
  loader's error and remedy instead of the page content.
  **Updated 2026-10-25:** the as-of date, the warning, the disclaimer and the
  limitations are no longer shown (D29). The error and remedy for a missing or
  altered artifact are unchanged.
- **Local only:** `.streamlit/config.toml` binds the server to `localhost`
  and sets `browser.gatherUsageStats = false`. Streamlit reads that file only
  from the working directory, so a launch from elsewhere would listen on all
  interfaces with statistics on; the app therefore checks both settings at
  startup and refuses to show anything until it is relaunched from the
  project root. The app writes nothing.
- **Workflow integrity:** the workflow page wraps the market and distress
  tools with the D19 manifest check (`loaders.with_frozen_checks`), so it
  refuses the same altered outputs as the other pages. The CLI is unchanged.
- **Dependencies:** the `app` extra includes the `rag` and `agent` extras,
  because the app imports both.
- **Caching:** loader results are cached on a fingerprint (size and
  modification time) of the files they read, so a changed artifact is loaded
  and verified again. The index and embedding model, and the workflow tools,
  are cached as resources; the workflow is re-created every 10 minutes so a
  restarted Ollama is picked up.
- **Charts:** checked by rendering every chart and page with headless Chrome
  in light and dark mode. Fixes from that review: legends below the plot,
  reference lines named in the legend instead of written over the data,
  wrapped labels, readable confusion-matrix cells, a date axis on the score
  timeline, and dark-mode colors (`charts.themed`, the palette's dark steps).
- **Tests:** `tests/test_app.py` runs the pages with Streamlit's AppTest (no
  server): every page renders with its notices and limitations, missing and
  altered artifacts are reported, and the RAG and workflow pages give the
  same answers as the pipeline and the CLI. (Updated 2026-10-25: pages are
  now checked to render without the notices and limitations, D29.)

---

## D22. Integrity verification of all frozen inputs and outputs (Phase H, step 1)

- **Date:** 2026-10-25
- **Problem:** the integrity checks run after each step used hash files kept
  outside the repository. The raw market snapshot, the Taiwan raw and
  processed data, and the RAG validation questions had no committed check.
- **Decision:** `src/fincrisis/qa/integrity.py`, run by
  `scripts/verify_integrity.py` (exit code 1 on any failure), checks files
  only against checksums recorded independently of them:
  - every entry of `config/frozen_artifacts.json` (D19);
  - the raw S&P 500 and VIX files against `raw_sha256` in the frozen
    `selection.json`, which the manifest covers, and against the snapshot's
    own metadata. The first catches an edit even if the metadata is edited
    to match;
  - the Taiwan raw `data.csv` and processed `data.csv`, `column_map.csv`
    and `split.csv` against the `input` checksums in the frozen
    company-distress `test_metrics.json`, which the manifest covers;
  - the RAG index against its manifest, the configured model and settings,
    and the current documents.
- **RAG validation questions:** added to the manifest. The checksum
  (`7ed9c16f...`) was recorded at 19:55:49 on 2026-10-24, before the set was
  first used. The file was committed once, in `66d8040`, and that commit's
  blob has the same checksum; a test checks this against Git history.
- **Line endings:** Git on Windows (`core.autocrlf=true`) checked the file out
  with CRLF line endings in a fresh clone, which changes its checksum.
  `.gitattributes` now fixes LF endings for `knowledge_base/eval/*.json` and
  the manifest, so every checkout has the recorded bytes.
- **Tests:** `tests/test_integrity.py` alters, removes or corrupts each kind
  of file on temporary copies and checks that the verifier reports it.

---

## D23. Reproducibility (drift) checks (Phase H, step 2)

- **Date:** 2026-10-25
- **Problem:** the dashboard and the workflow recompute features and scores
  with today's code. If the feature or target code, the saved models' behaviour
  or a library changed, they would silently diverge from what was evaluated.
- **Decision:** `src/fincrisis/qa/drift.py`, run by
  `scripts/check_reproducibility.py` (exit code 1 on any difference),
  recomputes values that the frozen runs saved and compares them:
  - the 16 market features and the target, rebuilt from the frozen snapshot,
    against `dataset.csv` (exact);
  - the saved market-stress models and the rule scores on the 2,950 held-out
    test days against `test_predictions.csv`;
  - the saved company-distress models on the 1,364 held-out test companies
    against their `test_predictions.csv` (same rows and labels, in order);
  - the latest-day signal from `assess_market` against the same computation
    done from the frozen pieces directly, including the threshold and signal.
- **Tolerance:** model scores within 1e-6. The prediction files store float32
  outputs as text, so XGBoost reproduces them to about 3e-8; anything larger is
  drift. Features and the target must match exactly.
- **Not an evaluation:** nothing is fitted, and no metric is computed. The test
  rows are re-scored only to compare scores with the saved ones; the models
  are loaded only after their checksum check.
- **Tests:** `tests/test_drift.py` simulates each kind of drift in memory (a
  feature changed by one part in 10^12, a changed target threshold, models
  shifted by 1e-4, reordered test rows, a diverging latest-day score) and
  checks that it is reported, and that float32-sized differences are not.

---

## D24. Guard on the RAG validation set (Phase H, step 3)

- **Date:** 2026-10-25
- **Problem:** the rule that the validation set is used once and never for
  tuning (D16) was only a convention. `scripts/evaluate_rag.py` and
  `scripts/evaluate_rag_generation.py` ran on it whenever given its path, and
  neither script had tests.
- **Decision:** both scripts resolve their question set through
  `src/fincrisis/qa/validation_guard.py` before touching the index, the
  embedding model or the Ollama server:
  - the development set (`rag.eval_set`) is the default, as before;
  - the validation set is recognised by its path or by its content (a
    renamed copy, with either line ending, is recognised too) and refused
    unless `--final-validation` is given;
  - with `--final-validation`, it must still match its checksum in
    `config/frozen_artifacts.json` (D22), and the flag is refused for any
    other question set;
  - if the recorded checksum is unavailable, evaluation is refused rather than
    allowed.

  A refusal prints the reason and exits with code 2; nothing is written.
- **Not covered:** an edited copy of the validation set no longer matches its
  checksum and would not be recognised. The guard prevents accidental reuse,
  not deliberate circumvention.
- **Tests:** `tests/test_eval_scripts.py` runs both scripts on a synthetic
  corpus in a temporary folder (fake embeddings and a fake local model): the
  development-set reports, keeping an existing report, an invalid question
  set, Ollama unavailable, and the refusals, which happen before the index or
  Ollama is touched. The project's reports are not written.

---

## D25. RAG regression checks and reference answers (Phase H, step 4)

- **Date:** 2026-10-25
- **Problem:** documents change at every step, and every edit changes the
  index. Nothing detected a retrieval regression or a changed answer; the
  before/after comparisons so far were manual.
- **Finding:** development-set retrieval has drifted slightly since D16 as the
  index grew from 201 to 247 passages with the Phase F-H documents. For the
  configured dense retriever, hit@3 went from 0.538 to 0.500, hit@5 from
  0.615 to 0.577 and MRR from 0.362 to 0.344 (each one question out of 26;
  q01 and q12 left the top five, q19 entered it, q08 and q22 fell one place).
  Abstention is unchanged. The first run of this check then caught the
  next step: adding this record moved development question q19 from fifth
  to sixth place, behind a D24 passage in a near-tie
  (cosine 0.293 against 0.291), so hit@5 fell to 0.538 and MRR to 0.337.
  That is corpus growth, not a setting change; it was reviewed and accepted.
  Nothing was tuned in response.
- **Decision:** `src/fincrisis/qa/rag_regression.py`, run by
  `scripts/check_rag_regression.py` (exit code 1 on a regression), compares
  the current index and code with two baselines in `config/qa/`:
  - `rag_dev_baseline.json`: development-set retrieval (summary metrics and
    each question's first relevant rank) for all three retrievers. For the
    configured one, no metric may fall below the baseline; the others are
    reported only. The file also keeps the D16 numbers for comparison.
  - `reference_answers.json`: the extractive answers of `ask_rag.py` for
    three questions and of `run_crisis_agent.py` for five, compared exactly
    (answer, citations, route). The stale-snapshot warning, which depends on
    today's date, is left out.
- **Development set only:** the question file is resolved through the
  validation guard (D24). Every fixed question is taken from the development
  set or the README, which a test checks, so none can be a validation
  question.
- **Accepting a change:** after a legitimate change (a new document or
  decision record), review what the check reports, then re-record with
  `--update`. The baselines record results; they do not choose settings.
- **Tests:** `tests/test_rag_regression.py` (comparison logic, the
  development-set rule, the update path, a stale index, and the check on the
  real index when it is current).

---

## D26. Ollama failure handling tests (Phase H, step 5)

- **Date:** 2026-10-25
- **Problem:** the local model was tested only with a fake transport on the
  happy path. Timeouts, HTTP errors, malformed or empty replies, a server
  that disappears, and how these reach the workflow and the dashboard had no
  tests.
- **Finding:** the code already handled them. A failed or rejected
  generation falls back to the extractive answer with a warning: `answer()`
  catches any generator error, and an empty reply is rejected by the
  grounding check. No code change was needed.
- **Tests:** `tests/test_ollama.py`:
  - a stand-in HTTP server on 127.0.0.1 plays Ollama, so the real `urllib`
    code runs. HTTP 500, malformed JSON, a reply slower than the timeout,
    an empty reply and a closed port each give the extractive answer with
    its citations and a warning, and the timeout is honoured. A malformed
    model list makes `ollama_status` report the model unavailable without
    raising;
  - the workflow wiring with the model unavailable (noted once, answers
    extractive) and with generation failing (the warning reaches the
    workflow's warnings);
  - the RAG and workflow pages with the toggle on: an unavailable model and
    a failed generation are shown as warnings next to the extractive answer;
  - an opt-in live test against the configured local model, run only when
    `FINCRISIS_LIVE_OLLAMA=1` is set. It checks structure and that an
    accepted answer passes the grounding check, not the wording.

---

## D27. Script tests and dashboard rerun tests (Phase H, step 6)

- **Date:** 2026-10-25
- **Problem:** `build_rag_index.py`, `ask_rag.py`,
  `download_taiwan_bankruptcy.py` and `inspect_taiwan_bankruptcy.py` had no
  tests, and the dashboard was not tested across reruns, with a stale
  snapshot, without the embedding model, or as a running server.
- **Tests:**
  - `tests/test_scripts.py` runs each script in a temporary project folder:
    - building the index, refusing to overwrite without `--force`, a
      missing embedding model and no documents (nothing written);
    - `ask_rag.py`: an answer with sources, `--json`, abstention with the
      threshold shown, an unavailable local model, and a stale index
      (exit code 1);
    - the Taiwan download with the network replaced by an in-memory zip:
      checksums recorded, only the named archive member read (an archive
      member `../evil.txt` is not extracted), an existing download never
      touched without `--force`, invalid downloads writing nothing;
    - the Taiwan inspection on a copy of the raw data: reports written, raw
      data unchanged, no overwrite, an altered dataset refused.
  - `tests/test_app_reruns.py`: a frozen output altered after a page has
    loaded is refused on the next rerun; the stale-snapshot warning appears
    on the pages (updated 2026-10-25: now that no page shows the snapshot
    notice, even when the snapshot is stale, D29); the RAG page explains a
    missing embedding model; and an
    opt-in live test (`FINCRISIS_LIVE_SERVER=1`) starts the real server from
    the project root, checks that it listens only on loopback addresses and
    answers, and then stops the whole process tree.
- **Found while testing:** on Windows the virtual environment's `python.exe`
  starts the real interpreter as a child process. Stopping only the
  launcher can leave the server running, and a dashboard server from an
  earlier session had kept running on 127.0.0.1:8501 that way. It was
  stopped, and the live test now stops the whole process tree.

---

## D28. One QA command (Phase H, step 7)

- **Date:** 2026-10-25
- **Decision:** `scripts/run_qa.py` (logic in `src/fincrisis/qa/runner.py`)
  runs five read-only stages and prints one pass/fail line for each; exit
  code 1 if any fails:
  1. **environment** (`src/fincrisis/qa/environment.py`, new): every package
     in `requirements-lock.txt` is installed at exactly its locked version.
     This names the cause when a library change alters results (which D23
     would detect). Installed packages that are not locked are listed only;
  2. **integrity** (D22);
  3. **reproducibility** (D23);
  4. **rag regression** (D25);
  5. **tests**: the unit-test suite, in a separate process.
- **Options:** `--quick` skips the test suite; `--live` also runs the live
  Ollama (D26) and dashboard-server (D27) tests; `--json` prints every check.
  A stage that crashes fails the run without hiding the other stages.
- **State at this decision:** all 182 locked packages are installed at their
  locked versions, with nothing unlocked; every stage passes.
- **Tests:** `tests/test_qa_runner.py` (the environment check on a synthetic
  lock file, parsing the test summary, the live switches, a failing or
  crashing stage, and a quick run on the project). The test stage is checked
  with a stubbed process, so the suite never runs itself.
- **Phase H summary:** integrity (D22), reproducibility (D23), the
  validation-set guard (D24), RAG regression (D25), Ollama failures (D26),
  scripts and dashboard reruns (D27), and this command (D28). The test suite
  grew from 224 to about 300 tests, and nothing was retrained, re-evaluated
  or tuned.

---

## D29. Simpler dashboard presentation (after Phase H)

- **Date:** 2026-10-25
- **Decision:** the dashboard pages were made shorter and easier to read for
  people without a finance or machine-learning background. Only presentation
  changed: no data, model, threshold, frozen output or check changed, and the
  CLI scripts are unchanged.
- **Removed from the pages:**
  - the snapshot's as-of notice and its stale-snapshot warning (page
    headers), and the snapshot date in the sidebar;
  - the research-only disclaimer and every "Limitations" section;
  - the Market Dashboard's "Snapshot details" section and the RAG
    Assistant's index line (passages, build time, embedding model).
  - The workflow page drops the disclaimer that the graph appends to each
    answer (D17), also from its "Full workflow state" view;
    `scripts/run_crisis_agent.py` still prints it.
- **Still shown:** a missing or altered artifact still shows the loader's
  error and remedy (D18, D19), and a missing or out-of-date index still
  disables RAG questions. After a market question, the workflow's answer
  still warns when the snapshot is old.
- **Trade-off:** the interface no longer states the age of the market data,
  warns when it is stale, or lists the limitations of each component. The
  limitations remain in `docs/model_card.md`, the frozen reports (also shown
  on the pages as "Final report" and "Validation report") and this file.
  `disclosures.py` still builds the notices, so they can be restored.
- **Reworded:**
  - shorter introductions on the Market Dashboard, Market Stress Analysis,
    Company Distress and RAG Assistant pages; simpler labels for the
    local-model toggles;
  - the Market Stress signal summary says whether the primary model's latest
    score is below or at/above its frozen threshold, and gives only its test
    recall at that threshold (as a share of the days followed within 20
    sessions by a 5% fall, D2). Average precision and precision are no longer
    in that paragraph; the evaluation charts below still show them;
  - the Company Distress page shows a plainer wording of the label caveat.
    The frozen `label_caveat` in the results files is unchanged, and the new
    wording still makes no claim that label 1 means bankruptcy (D4).
- **Added to the Market Dashboard:**
  - a plain-language explanation in muted red beneath each of the four
    indicators, with its example filled in from the latest values;
  - historical events shaded on both market-history charts, labelled with
    their name and dates, with context on hover. Each event in
    `charts.MARKET_EVENTS` runs from an S&P 500 closing high to the following
    closing low. The 2007–09 financial crisis (2007-10-09 to 2009-03-09) and
    the 2020 COVID-19 crash (2020-02-19 to 2020-03-23) were checked against
    published index history and the frozen snapshot; a test checks every
    event's dates against the snapshot, so events added later are checked too;
  - a "Market overview summary" at the end of the page
    (`disclosures.market_overview`). It is built from the snapshot without a
    language model. Each sentence places the latest reading against the
    snapshot's own history (bottom, middle or top third), and the drawdown
    sentence uses the usual 10% "correction" and 20% "bear market" terms. A
    sentence whose inputs are missing is left out.
- **Charts:** every chart hides the Zoom, Pan, Autoscale and Reset axes
  toolbar buttons. Hover, legends and the other buttons are kept, and
  dragging on a chart still zooms.
- **Not checked visually:** unlike the D21 review, these changes were not
  rendered in a browser; the image exporter is not installed, and adding it
  would fail the environment check (D28). They are covered by AppTest and
  unit tests only.
- **Tests:** `tests/test_app.py` and `tests/test_app_reruns.py` check that no
  page shows the notices, limitations or disclaimer, even with a stale
  snapshot. They also check the workflow page answer (the CLI answer without
  the disclaimer), the indicator explanations, the summary's position and
  text, and the toolbar setting on every chart. `tests/test_dashboard.py`
  covers the summary's wording and missing inputs; `tests/test_charts.py`
  covers the event shading, labels and dates.
- **RAG reference answers (D25):** adding this record changed one of the
  three extractive reference answers in `config/qa/reference_answers.json`
  (the one citing D2). Its two sentences from D2 and their citation are
  unchanged; its off-topic sentence from D20 is replaced by an off-topic one
  from this record. The configured (dense) retriever's metrics did not
  change. The lexical retriever, which is only reported, moved slightly since
  the D25 recording (251 to 270 passages): hit@1 0.308 to 0.346, MRR 0.496
  to 0.515, with q01 falling from first to second place. Both were reviewed
  and re-recorded with `--update`.

---

## D30. Structure-aware, token-bounded chunking

- **Date:** 2026-10-25
- **Problem (development-set audit):** the character splitter (D14) cut
  800-character chunks without regard to their content, and produced three
  problems:
  - **Cut off by the model:** 53 of 270 chunks (20%) were longer than the 256
    tokens `all-MiniLM-L6-v2` reads, and the model ignores the rest. The
    model's limit includes its two special tokens; this was checked by
    embedding a 256-token text with and without words appended.
  - **Split content:** the D2 target formula was split from its definition,
    so the D2 chunk that ranked first held no usable sentence. Tables lost
    their header in later chunks.
  - **Fragments:** tiny scraps (the end of a list, a report title) ranked
    near the top for many questions.

  The splitter also stripped indentation, including inside code blocks, and
  flattened nested lists.
- **Decision:** `src/fincrisis/rag/chunking.py`, selected with
  `rag.chunking.method: structured`:
  - **Sections:** split by h1-h3 headings, with the same section paths as
    before, and every line kept as written. Each chunk records the source
    lines it covers (`start_line`, `end_line`).
  - **Packing:** whole blocks (paragraphs, list items, tables, code blocks)
    are packed into chunks of at most the model's `max_seq_length`, counted
    with its own tokenizer on the exact text it embeds, special tokens
    included (and the section path, when `embed_section_headings` is on).
  - **Kept together:** a heading or a lead-in ending in ":" (also when the
    colon is inside emphasis, as in `**Details:**`) stays with the block that
    follows it, and a block that starts lowercase (a sentence continued after
    a formula) stays with the block before it.
  - **Splitting a long block:**
    - a table is split between rows, spread evenly; every part repeats the
      header, and later parts also repeat the section heading;
    - a code block is split between lines and re-fenced;
    - prose is split between sentences, and only a single over-long sentence
      between words.

    The first part is sized to fit with the heading or short introduction
    before it.
  - **Short pieces:** a trailing chunk under `min_tokens` (64, fixed before
    measuring) is merged into the chunk before it. A parent section with
    almost no text (a title) is merged into the first chunk of its first
    subsection.
  - **Manifest:** records the chunking. An index built with other chunking
    is refused, as for `embed_section_headings` (D16). Settings without a
    `chunking` block, and older indexes, mean the original splitter, which
    `method: characters` still selects.
- **Unchanged:** the relevance threshold, `top_k`, the retriever, sentence
  selection, the generation prompt and the grounding checks.
- **Results (development set only; the validation set was not used):**
  - **Index:** 270 chunks (53 over the limit) became 288 chunks, none over
    the limit. The 4 chunks under 64 tokens are complete short sections.
  - **Configured dense retriever** ("After": the documents as they were;
    "With this record": once this entry is indexed too, 297 chunks, with the
    lead-in fix below):

    | Measure | Before | After | With this record |
    |---|---|---|---|
    | hit@1 | 0.192 | 0.269 | 0.269 |
    | hit@3 | 0.500 | 0.577 | 0.577 |
    | hit@5 | 0.538 | 0.692 | 0.654 |
    | MRR | 0.337 | 0.435 | 0.428 |
    | Refusals on unanswerable questions | 0.500 | 0.500 | 0.500 |

  - **Reported-only retrievers:** fell. Lexical hit@1 went from 0.346 to
    0.308 and MRR from 0.515 to 0.501. Hybrid hit@3 went from 0.692 to 0.615
    and hit@5 from 0.808 to 0.731.
  - **Audit probes:** the D2 passage rose from ranks 44 / 5 / 13 to
    4 / 1 / 2 for the three probes that need it. The 16-feature table
    and the numbers-only rows of the test-results table still rank poorly;
    that is left to retrieval (a later step).
- **Regression found and fixed (lead-in rule):**
  - **Symptom:** the first version made the workflow's market explanation
    (D17) quote D13's secondary-metric rule instead of its primary-model
    selection rule.
  - **Cause:** the lead-in rule tested for a trailing ":", but this project
    usually puts the colon inside bold (`**Label:**`), so 34 lead-ins were not
    kept with their lists. D13's "Primary model and test-set safeguards…:**"
    ended one chunk while its first item, the selection rule, started the
    next. Cut off from its introduction, that chunk scored 0.288 against the
    0.30 per-passage threshold of scoped retrieval and was dropped, although
    the sentence alone scores 0.80.
  - **Fix:** a lead-in may end in ":" followed by closing `*` or `_`. The D13
    chunk then starts with its lead-in and scores 0.327, and the workflow
    quotes the selection rule again. The dense retriever's development
    results did not change.
  - **Not changed:** the 256-token limit, the threshold, `top_k` and
    sentence selection.
- **Considered:** a 200-token cap. On the development set it raised hit@1 to
  0.346 and MRR to 0.469, kept the D13 passage, and found the test-results
  row, but lowered hit@3 to 0.538 and hit@5 to 0.654. One question moves
  these rates by about 0.04, so the difference is within noise. Packing to
  the model's limit was kept. A size target should be compared only on a
  new held-out question set.
- **RAG regression baselines (D25):** re-recorded after review.
  - Unchanged in substance: the VIX answer (citation ids renumbered) and the
    out-of-scope refusal (top score only).
  - Better: the label-horizon answer now starts with D2's definition.
  - Unchanged: the workflow market answer, once the lead-in fix restored
    the selection rule.
- **Tests:** `tests/test_chunking.py` covers:
  - token limits, including the section prefix;
  - split tables, re-fenced code, and definitions kept together;
  - merged fragments, no heading-only chunks, complete short sections kept;
  - citation ids, sections and line ranges;
  - prose and sentence boundaries;
  - refusing an index with other chunking;
  - on the project documents with the real tokenizer: no chunk over the
    limit, D2 in one chunk, and the development questions still valid;
  - lead-ins with the colon inside emphasis, and that no project chunk ends
    with a lead-in whose list continues in the next. Both tests fail with the
    old rule.

  `tests/test_agent.py` checks that the workflow's D13 explanation states
  the selection rule: the primary model has the highest pooled walk-forward
  average precision. It checks the meaning, not exact wording.

---

## D31. Wider retrieval with a cross-encoder reranker

- **Date:** 2026-10-25
- **Problem:** after D30, the passage that answers a question was often
  ranked 5th to 15th by dense search, below passages that only share its
  topic. Only the top 4 are used.
- **Decision:** retrieval mode `rerank` (`rag.retrieval.mode`), now the
  default:
  - dense search returns the top `rerank_candidates` (20, fixed before
    measuring);
  - a local cross-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2`, reads the
    question with each candidate and re-orders them; the top `top_k` (4) are
    the evidence;
  - the reranker reads each passage after its citation location
    (`source - section path`). Without it, the held-out test and validation
    reports, whose tables have the same layout, could not be told apart:
    text-only reranking cited a validation row for a test-set question.
- **Unchanged:**
  - the refusal gate is still the best dense cosine over all chunks against
    `min_relevance` (0.30), so refusals on the development set are
    unchanged;
  - as in the D16 hybrid mode, passages below 0.30 are no longer dropped
    one by one once the gate is passed;
  - `top_k`, sentence selection, the prompt, the grounding checks, the
    index, and the workflow's section-scoped context (D17), which stays
    dense.
- **Model:** Apache-2.0, 22.7M parameters, CPU. It was trained on MS MARCO,
  like the embedding model. It is loaded from the local cache only;
  `scripts/build_rag_index.py --allow-download` caches it once. If it is
  missing, the RAG page and the workflow report a clear error.
- **Results (development set only; the validation set was not used):**

  | Mode | hit@1 | hit@3 | hit@5 | MRR | Refused (unanswerable) |
  |---|---|---|---|---|---|
  | dense (D30) | 0.269 | 0.577 | 0.654 | 0.428 | 0.500 |
  | rerank, passage text only | 0.385 | 0.538 | 0.731 | 0.497 | 0.500 |
  | rerank with source and section (adopted) | 0.423 | 0.692 | 0.846 | 0.573 | 0.500 |
  | the same, after the records fix below | 0.462 | 0.731 | 0.846 | 0.596 | 0.500 |

  The section variant was the only variant tried after the text-only run,
  chosen for the reason above. The audit's model-comparison questions now
  cite the held-out test rows; the 16-feature list and the test results
  table are still not retrieved. When the right passage ranks first, the
  answer can still quote sentences from the other evidence passages: that
  is sentence selection, the next planned step.
- **Records quoting evaluation questions (found and fixed):**
  - **Problem:** D25, D29 and D30 quoted or paraphrased development and
    reference questions while discussing the RAG's own evaluation. The
    reranker ranked those passages highly for exactly those questions, so a
    reference answer started with an off-topic sentence about the
    evaluation. This breaks the existing rule that the RAG's own reports are
    not indexed because they quote the evaluation questions.
  - **Rule:** indexed documents refer to evaluation questions by id (for
    example q19) or by the decision they concern, and do not quote them.
    The README is exempt: D25 takes reference questions from its examples.
  - **Changes:** the three passages in D25, D29 and D30 were reworded on
    2026-10-25. Their facts are unchanged.
  - **Test:** `tests/test_rag_regression.py` fails if any other indexed
    document contains the first six words of a development or reference
    question. A paraphrase is not caught, so the rule also applies when
    writing records.
- **Caution:** like D16, these gains are measured on the development set,
  which has been used to diagnose the system. They should be confirmed on a
  new held-out question set before further tuning.
- **Tests:** `tests/test_rerank.py` covers:
  - candidates are the dense top, re-ordered by the reranker;
  - the reranker reads source and section;
  - scores stay dense cosines, and a high rerank score cannot pass the gate;
  - invalid settings are refused, and a missing model gives a clear error;
  - the evaluation modes, and a configured mode without a baseline must be
    reviewed;
  - the real model ranks a relevant passage above an unrelated one.

---

## D32. Extractive sentences chosen by the reranker, continued in context

- **Date:** 2026-10-25. **Status:** current implementation, measured on the
  development set only. The RAG baselines (D25) were re-recorded for it, and
  for D33, after review on 2026-10-25.
- **Problem (audit):** the extractive answer picked the sentences most
  similar to the question by embedding. That favours sentences that repeat
  the question's words over sentences that answer it: the D2 definition
  lost to a sentence about what trading days are.
- **Decision:**
  - **Scoring (`rag.sentence_selection: reranker`):** the cross-encoder of
    D31 scores every candidate sentence of the evidence passages, each read
    after its passage's `source - section path`, the format it reads for
    passages.
  - **Continuation (`rag.sentence_continuation: all`):**
    - the best-scoring sentence opens the answer;
    - the sentences that follow it in its own passage come next, whatever
      their scores;
    - the next-best sentences of all passages fill the rest, up to
      `max_answer_sentences` (3);
    - each sentence is cited to its passage.

    Scored on their own, the steps and qualifications that complete a rule
    lose to sentences that repeat the question's words. Continuation keeps
    them.
  - **Variants:** one at each stage, decided before measuring. There are no
    weights.
  - **Scope:** mode `rerank` only. The workflow's section-scoped context
    (D17) keeps embedding selection.
- **Unchanged:** the refusal gate, `top_k`, the evidence passages, the
  candidate sentences (`answer.sentences`), the citations, the prompt, the
  grounding checks and the generated-answer path. `sentence_selection:
  embedding` restores the D14 behaviour.
- **How the design was reached:**
  1. **Score order alone** (the first version of this decision): a
     test-requirement sentence from D11 repeats nearly every word of the
     VIX-timing questions. It replaced D11's alignment step about VIX-only
     dates in the VIX reference answer and in q05.
  2. **Continuation** fixed both, and is kept.
  3. **Relevance boundary** (one approved experiment, rejected): continue
     only with sentences scored above 0, the cross-encoder's own boundary.
     - **Why it failed:** read out of context, the cross-encoder scores many
       useful continuation sentences far below 0. Among them are the
       bootstrap method for q24, the licence line for q19, the yfinance
       caveats for q04, and D11's alignment step (-1.30).
     - **Effect:** it brought the VIX regression back, and raised the
       irrelevant sentences in the changed answers from 14 to 22 by manual
       count.
     - **Status:** the code remains (`sentence_continuation: relevant`) but
       must not be used.
- **Results (development set only, 26 answerable questions; the validation
  set was not used):**

  | Measure | Embedding (D14) | Score order | Continuation (kept) | Boundary (rejected) |
  |---|---|---|---|---|
  | Answer contains an expected phrase | 8 | 11 | 12 | 11 |
  | First sentence contains it | 3 | 8 | 8 | 8 |
  | Answer cites a relevant passage | 16 | 17 | 14 | 17 |

  - **Refusals:** unchanged in every variant. Sentence selection does not
    affect the gate; 3 of the 6 unanswerable questions are refused.
  - **Manual review, continuation against score order** (all changed
    answers): 11 better (q01, q04, q05, q08, q09, q11, q16, q18, q19, q24,
    q25), 5 worse (q02, q06, q07, q17, q23), the rest equal.
  - **Gains over score order:**
    - q18 now gives the reason (`row_id 2499`);
    - q24 now names the circular block bootstrap;
    - q19 now includes the licence line;
    - q11 now says the label meanings remain unverified.
  - **Lower citation count:** "cites a relevant passage" falls because
    answers now draw more on one passage.
- **Known regressions (continuation, the current implementation):**
  - **q06:** loses its answer. Its best sentence (why VIX is lagged) is
    followed in its passage by lag reasoning, which displaces D11's
    alignment step that score order took from another passage.
  - **q02, q07 and q23:** continuation runs into unrelated sentences of the
    same passage: a disclaimer, a row count, "Yahoo Finance data is
    unofficial".
  - **q17:** slightly worse; it drops the sentence stating that 24 features
    contain values above 1.
  - **q22:** not answered. The model-files lead-in sits in a chunk that
    ranks 30th, outside the 20 reranker candidates. That lead-in also ends
    in ":", so it is not a candidate sentence. Embedding selection had
    included the path only through a bare file-path line from another
    section. This is a retrieval and segmentation problem. D33 fixed the
    retrieval half; the rest is accepted as a documented limitation for
    now.
  - **VIX reference answers (RAG and workflow):** D11's full procedure
    (rule, alignment step naming the VIX-only dates 2026-05-25 and
    2026-09-07, shift) replaces the dataset report's line on VIX-only
    dates. The answers now cite D11 only; the dataset report is no longer
    cited.
- **RAG reference answers (D25):** reviewed and re-recorded with exactly
  three differences. Status, refusals and routes are unchanged, and the
  other five reference answers are identical.
  - **VIX reference answer (RAG):** the rule, D11's alignment step and the
    shift; cites D11 only, as above.
  - **The same answer from the workflow:** the same change.
  - **Label-horizon answer (the one citing D2):** opens with the
    definition, followed by D2's two sentences on the forward running peak.
    It now cites one D2 passage instead of two.
- **Other weakness:** table header rows count as candidate sentences and
  are sometimes chosen (q07). Candidate filtering was not changed.
- **Caution:** measured on the development set only, like D30 and D31.
- **Tests:** `tests/test_rerank.py` covers:
  - the scorer chooses the sentences and the citations follow;
  - each sentence is read after its passage's source and section;
  - continuation after the best sentence, and the fallback when nothing
    follows it;
  - the rejected boundary variant, and that the setting is validated and
    configured as `all`;
  - without a scorer, embedding selection is unchanged;
  - section-scoped context does not use the scorer;
  - the scorer is attached only when configured, in rerank mode;
  - with the real model, the D2 definition outscores the sentence that
    repeats the question.

---

## D33. Hex digests read as a placeholder by the models

- **Date:** 2026-10-25. **Status:** current implementation, measured on the
  development set only. The RAG baselines (D25) were re-recorded after
  review together with D32; masking itself changed no reference answer.
- **Problem (q22, D32):** the model-card passage stating where the
  company-distress model files are saved is 55% SHA-256 digests by tokens
  (137 of 251): three 64-character checksums in a table. The digests
  swamp its embedding, so it scored 0.320, below the 20th dense candidate
  (0.359), ranked 30th, and never reached the reranker.
- **Decision (`rag.retrieval.mask_digests: true`):** hex strings of 32 or
  more characters are replaced by `<sha256>` in what the models read:
  - the text embedded for the index;
  - the passages the reranker reads (D31);
  - the sentences it scores (D32).

  The stored chunk text, the evidence, the citations and the grounding
  checks keep the real digests, and an answer quotes them as written.
  - **Kept:** short git ids such as `1a96172` are not masked.
  - **Manifest:** records the setting. An index built with the other
    setting, or before D33 (treated as unmasked), is refused, as for
    `embed_section_headings`.
  - **Token counts:** D30 chunking still counts the unmasked text, which is
    longer, so no chunk can exceed the model's limit.
- **Not done (held):** keeping colon-terminated lead-ins that introduce a
  table or code block as candidate sentences. Simulated with this change,
  q22 was still not answered: the lead-in scores -0.62 against 3.70 for a
  sentence of the neighbouring passage, which opens the answer and is
  continued (D32). It would also add about 15 table captions as
  candidates.
- **Exposure:** 6 of the 309 chunks contain digests (median 34% of their
  tokens).
- **Results (development set only; the same 312 chunks unmasked and
  masked; the validation set was not used):**
  - **Configured rerank mode:** retrieval unchanged (hit@1 0.462, hit@3
    0.731, hit@5 0.846, MRR 0.596); so are the lexical and hybrid modes.
  - **Dense mode (reported only):** q08 moves from rank 3 to 4, so hit@3
    goes from 0.577 to 0.538 and MRR from 0.428 to 0.424.
  - **Answers:** no development answer and no refusal changes.
  - **q22:** the model-files passage now reaches the evidence (third of
    four), but the answer is unchanged. Its lead-in is not a candidate
    sentence, and the answer continues from the first passage (D32). q22
    remains a known regression until the held lead-in change, or another
    approved design, is evaluated.
- **Risk:** dense and reranked search can no longer match a digest's value;
  lexical and hybrid search, which read the raw text, still can.
- **Tests:** `tests/test_digests.py` covers:
  - which strings are masked (SHA-256, MD5, full git hashes, upper case)
    and which are kept (short git ids, 31 characters);
  - the embedded text is masked while the stored text keeps the digest;
  - the reranker and the sentence scorer read masked text, and the evidence
    and the answer keep the digest;
  - the setting reaches the retriever;
  - an index built with the other setting, or before D33, is refused by
    `open_store` and reported by the dashboard;
  - on the project index, the model-files passage reaches the reranker's
    candidates.

---

## D34. Audit of the market-stress dataset, splits and evaluation (Phases B and C)

- **Date:** 2026-10-25. **Status:** findings recorded and accepted as
  findings. This is **not** a final approval of Phase B or Phase C, and it
  changes no data, code, model, threshold or result.
- **Scope:** read-only review of D2, D3 and D10-D13 and their code, tests and
  frozen outputs. Nothing was trained, scored or re-evaluated, and the
  held-out test predictions were not analysed.
- **Approval record:** the repository has no earlier Phase B audit or
  approval, and no record of the review that D13 requires between its two
  stages. Phase B is documented by D11 and D12, by
  `data/reports/market_stress/dataset_report.md`, and by tests.
- **Safeguards verified (code and tests):**
  - **Target** (`src/fincrisis/features/market_target.py`, D2): uses only
    closes from *t*+1 to *t*+20, with the running peak starting at *C_t*. The
    last 20 rows are NaN, never 0. Tested in `tests/test_market_target.py`.
  - **Features** (`src/fincrisis/features/market_features.py`, D11, D12):
    - S&P 500 windows are trailing, using closes up to *t*;
    - VIX is aligned to the S&P 500 calendar, then lagged one session;
    - `vix_rv_ratio` uses the previous session's realized volatility.

    Prediction time is after the close on *t*. Tested in
    `tests/test_market_features.py`, including that later prices and VIX
    closes dated *t* or later never change a feature at *t*.
  - **Splits** (`src/fincrisis/evaluation/splits.py`, D12):
    - 15 expanding walk-forward folds (2000-2014);
    - training ends 20 sessions before each validation year, and
      development ends 20 sessions before 2015-01-02;
    - a purge shorter than the horizon is refused.

    With purge = horizon = 20, no training label uses a close inside its
    validation year, and no development label uses a test-period close.
    Because the training windows only expand forward, no embargo is needed
    after a validation year. Tested in `tests/test_market_splits.py`.
  - **Fitting** (`src/fincrisis/models/market_stress.py`,
    `scripts/train_market_stress.py`):
    - models are cloned and fitted on each fold's training rows;
    - the scaler is inside the logistic-regression pipeline;
    - there is no imputation, no feature selection and no tuning;
    - the CV stage receives development rows only;
    - the final stage fits on development rows and only scores the test
      rows.

    Tested in `tests/test_market_models.py`.
  - **Frozen outputs:** `selection.json` and `test_metrics.json` were both
    produced from clean commit `7e80173`, at 15:27:56 and 15:30:03 UTC on
    2026-10-24. The test results record the SHA-256 of the selection they
    used, which matches the current file. Integrity (D22) and
    reproducibility (D23) checks cover them and pass.
- **Data roles:**
  - **Development:** 1990-12-31 to 2014-12-02, 6,029 eligible rows.
  - **Out-of-fold validation:** the 15 yearly blocks of 2000-2014, 3,753
    rows. There is no separate validation set for market stress.
  - **Purged:** 2014-12-03 to 2014-12-31, 20 rows.
  - **Held-out test:** 2015-01-02 to 2026-09-25, 2,950 rows.
  - **Unlabelled:** the last 20 snapshot rows, to 2026-10-23.
- **The 2015-2026 test set has been evaluated (2026-10-24) and is no longer
  untouched.** Its results are historical results of the frozen models. It
  must not be used to select, tune or compare new models or thresholds.
- **Limits of the evidence on test-set timing:** the outputs show one final
  run, made after the selection was locked, from clean committed code. They
  cannot show that the test set was never evaluated or inspected before
  that, for example while the evaluation code was developed in `06f4ed7`.
  This is unverified, not disproved.
- **Open methodological questions** (recorded, not resolved):
  - **Threshold transfer** (decided 2026-10-26 in D35). Each model's
    threshold is the max-F1 threshold on the pooled out-of-fold predictions
    of 15 fold models. It is applied unchanged to a different model,
    refitted on all development rows.
    - **Rules:** the score is the raw feature, so the threshold transfers.
    - **Logistic regression and XGBoost:** fold score scales differ widely.
      The median out-of-fold score ranges over folds from 0.014 to 0.675
      for logistic regression and from 0.010 to 0.708 for XGBoost. The
      final model's scale was not checked, and the fold models were not
      saved.
    - **Affected:** the secondary XGBoost signal shown by the dashboard and
      the workflow.
  - **Pooled out-of-fold aggregation.** Pooled metrics also reward score
    offsets between folds, not only ranking within a year.
    - **Evidence:** the dummy's constant score per fold (its training
      prevalence) is negatively correlated with the validation prevalence
      (Spearman -0.52), so its pooled ROC-AUC is 0.318.
    - **The choice changes the result:** the primary model was selected on
      pooled average precision (0.618 for `rule_downside_vol_20`). The mean
      of the defined yearly values ranks `rule_vix_pct_252` first (0.461
      against 0.429).
    - **Already recorded:** the primary-model rule was added to D13 after
      the CV results were seen, before the test run.
  - **Dependence from overlapping labels.**
    - **Size:** label autocorrelation is 0.94 at lag 1, 0.27 at 20, 0.19 at
      60 and 0.11 at 120 sessions. There are about 2.6-2.7 episodes a year:
      65 in development and 30 in the test period. Their median length is
      17-19 sessions.
    - **What the bootstrap covers:** the 60-session circular block bootstrap
      (D13) covers dependence within an episode. It does not fully cover
      the longer clustering of volatile regimes. The 20- and 120-session
      sensitivity intervals are similar.
    - **What it does not cover:** the variability that comes from refitting
      the models, and each block wraps from the last row back to the first.
    - **Training:** overlapping rows are not down-weighted.
- **Historical results and a future evaluation:**
  - **Historical results:** the frozen D13 results stand as historical
    results and are not re-run.
  - **New models or thresholds:** any new model, threshold or selection rule
    needs evaluation on data that has not been used. With the snapshot
    frozen at 2026-10-23, that means forward data collected after it. Each
    new day's label is known 20 sessions later.
  - **Expected yield:** at about 2.6 episodes a year, one year gives too few
    episodes to compare models. A forward period is suitable for
    monitoring, not for model selection, unless it runs for many years.
- **Recommendations from the audit analysis** (added 2026-10-26). These are
  the minimum changes that would address the open questions. They are
  **not approved and not applied**. Each needs the matching decision below,
  and none authorizes training.
  - **Threshold transfer:** for fitted models, either derive the final
    model's threshold from out-of-fold predictions inside its own training
    data (nested), or report fitted-model threshold metrics as not
    transferable and rely on ranking metrics only. The rule baselines need
    no change.
  - **Aggregation:** before any future comparison, declare one primary
    selection metric (the mean of yearly average precision, or the pooled
    value) and always report the other. The yearly mean is less affected
    by score offsets between folds.
  - **Dependence:** keep the block bootstrap. Add a 250-session block
    sensitivity check, and report episode counts with every metric.
    Weighting training rows by label uniqueness is a training-time option,
    not a leakage fix.
  - **Historical results:** keep the D13 results as historical results,
    labelled as such.
  - **Forward monitoring:** if wanted, run it under a protocol fixed in
    advance, with the models fixed.
- **Decisions still required** (not made here):
  - whether to formally approve Phase B and the D13 protocol on this
    evidence;
  - how thresholds of fitted models should be set for a final model
    (decided 2026-10-26 in D35);
  - which aggregation is the primary selection metric for any future
    comparison;
  - whether to address overlapping labels in training;
  - how to record reviews between stages for any rerun;
  - whether and how to set up a forward evaluation.

---

## D35. Decision thresholds are valid for the rule baselines only

- **Date:** 2026-10-26. **Status:** decided by the project owner. It resolves
  the threshold-transfer question of D34 and none of the others.
- **Decision:**
  - **Rule baselines** (`rule_downside_vol_20`, the primary model, and
    `rule_vix_pct_252`): their frozen max-F1 thresholds (0.0896 and 0.2937)
    stay valid. A rule's score is the raw feature value, so the threshold
    means the same for the fold models and the final model.
  - **Fitted models** (`logistic_regression`, `xgboost`): their frozen
    thresholds (0.17 and 0.0652) are **not transferable**. They were chosen
    on the pooled out-of-fold scores of 15 fold models, whose score scales
    differ widely (D34), and applied unchanged to a model refitted on all
    development rows, whose scale was not checked.
    - **Judged on:** ranking metrics (average precision, ROC-AUC), plus the
      Brier score and log-loss as measures of probability quality.
    - **Not used as evidence:** their threshold metrics (precision, recall,
      F1 and confusion matrices at those thresholds).
  - **Historical results:** the threshold metrics of the fitted models in
    `data/reports/market_stress/models/` (`cv_report.md`, `final_report.md`,
    `test_metrics.json`) stay as historical results, read with this caveat.
    Nothing is recomputed.
  - **Future training:** any decision threshold for a fitted model needs its
    own decision, with the procedure fixed before training. A pooled
    out-of-fold threshold from fold models must not be applied to a refitted
    model. D35 does not approve or schedule any training.
- **Not changed:**
  - no code, data, model, threshold value or result;
  - `selection.json` is unchanged;
  - nothing was retrained or re-evaluated;
  - the integrity and reproducibility checks are unaffected.
- **Known inconsistency, until a separately approved change:**
  - the workflow's market answer still reports the secondary XGBoost model
    as elevated or not at its frozen threshold (`src/fincrisis/agents/tools.py`,
    `graph.py`);
  - the dashboard's Market Stress page still shows that comparison
    (`app/views/market_stress.py`).

  Under this decision, that signal is not supported. The proposed change is
  to show the XGBoost score and its historical percentile instead.
- **Still open from D34:**
  - the primary aggregation metric;
  - overlapping labels in training;
  - formal approval of Phase B and the D13 protocol;
  - how reviews between stages are recorded;
  - forward evaluation.
