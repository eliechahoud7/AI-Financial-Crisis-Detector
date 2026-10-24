# Model Card: Company Distress Baselines (Milestone 4)

> **Label caveat.** The target column `bankrupt` (originally `Bankrupt?`) takes
> the values 0 and 1. **UCI does not document what 0 and 1 mean, and their
> meanings remain unverified.** Label 1 is treated as the class to detect only
> because it is the minority class (3.23% of rows). Nothing in this card shows
> that these models predict confirmed bankruptcy.

All figures below come from the project's generated reports and metadata:

- `data/reports/taiwan_bankruptcy/baselines/cv_report.md`, `cv_results.csv`,
  `selection.json`, `permutation_importance.csv`
- `data/reports/taiwan_bankruptcy/baselines/final_report.md`, `test_metrics.json`
- `artifacts/models/taiwan_bankruptcy/model_metadata.json`
- `data/interim/taiwan_bankruptcy/metadata.json` and `data/raw/taiwan_bankruptcy/metadata.json`

Design decisions are recorded in `docs/decisions.md` (D4, D8, D9).

---

## 1. Model overview and intended use

| Item | Value |
|---|---|
| Task | Binary classification: rank companies by their score for label 1 (the minority class) |
| Models | DummyClassifier (prior), Logistic Regression (`lr_quantile`), XGBoost (`xgb_large_as_missing`) |
| Status | Baselines with fixed settings; no hyperparameter tuning |
| Code version | Git commit `31988f7eb4782bc7ce0c078834171084c83a5d70` (no uncommitted changes at run time) |
| Evaluated | Cross-validation 2026-10-24T14:04:43Z; held-out test 2026-10-24T14:05:23Z |

**Intended use:** research and education within this project. The baselines
provide a reference point for later models and for studying the dataset's
open questions, such as the large feature values.

**Not intended for** any real lending, investment, credit, audit, regulatory,
or other decision about a company (see section 11).

## 2. Dataset

| Item | Value |
|---|---|
| Name | Taiwanese Bankruptcy Prediction, UCI Machine Learning Repository |
| DOI | 10.24432/C5004D |
| License | CC BY 4.0 |
| Source, as described by UCI | Taiwan Economic Journal, 1999–2009; "bankruptcy" defined by Taiwan Stock Exchange business regulations |
| Downloaded | 2026-10-24T13:09:24Z from `https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip` |
| Raw CSV SHA-256 | `67bf2e7c75490f7ad3f76bbce57d49cdc25967cdab607527b94f944863fa14d8` |
| Size | 6,819 rows, 96 columns (1 target + 95 features) |
| Class distribution | Label 0: 6,599 (96.77%); label 1: 220 (3.23%) |
| Missing values | None |

**Historical data.** The data covers 1999–2009 and Taiwanese companies only.
It may not reflect current financial conditions, other countries, accounting
standards, or market structures.

**Unknown or unverified:**

- The meanings of labels 0 and 1. UCI's documentation and our searches gave
  no authoritative coding. The original paper usually associated with the
  dataset (Liang et al., 2016) could not be accessed.
- Units and definitions of the features. UCI provides names only.
- Whether rows are one per company or several per company (firm-years). There
  is no company identifier.
- When each feature was measured relative to the event behind label 1.
- The cause of the large values in 24 features (section 5).
- Whether UCI's file is identical to the original source data.

## 3. Preprocessing and train/test split

Preprocessing is done by `scripts/preprocess_taiwan_bankruptcy.py` (D8). It
uses only deterministic steps that don't look at the target, and nothing is
fitted.

- **Exact parsing:** `float_precision="round_trip"`.
- **Column names:** converted to ASCII snake_case (`Bankrupt?` → `bankrupt`).
  Original names and positions are kept in `column_map.csv`.
- **Four columns dropped**, each relationship re-checked on the data at run time:

  | Original position | Original name | Reason |
  |---|---|---|
  | 38 | `' Net worth/Assets'` | equals 1 − `Debt ratio %` (max deviation 1.1e−15) |
  | 77 | `' Current Liability to Liability'` | identical to `Current Liabilities/Liability` |
  | 78 | `' Current Liability to Equity'` | identical to `Current Liabilities/Equity` |
  | 94 | `' Net Income Flag'` | constant (always 1) |

- **Kept unchanged:** `liability_assets_flag` (near-constant: 8 rows equal 1),
  `operating_gross_margin` and `gross_profit_to_sales` (near-identical, not
  identical), and all large-value features.
- **No values changed:** no clipping, scaling, imputation, or transformation.
- **Result:** 6,819 rows × 93 columns (`row_id`, `bankrupt`, 91 features).
- **Split:** stratified 80/20, seed 42, saved in `split.csv` and reused unchanged.

| Set | Rows | Label 0 | Label 1 |
|---|---|---|---|
| Train | 5,455 | 5,279 | 176 |
| Test | 1,364 | 1,320 | 44 |

Processed-file SHA-256: `data.csv` `835d395a…0a6c`, `split.csv` `7cc175d6…c9f0`,
`column_map.csv` `e3a94e5e…f90f` (full values in `selection.json`).

## 4. Evaluated baselines

| Family | Config | Pipeline | Purpose |
|---|---|---|---|
| Dummy | `dummy` | `DummyClassifier(strategy="prior")` | Lower bound: predicts the training share of label 1 for every row |
| Logistic Regression | `lr_quantile` | `QuantileTransformer(output_distribution="normal")` → `LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")` | Interpretable linear baseline |
| XGBoost | `xgb_large_as_missing` | `LargeValuesToMissing(threshold=1.0)` → `BalancedXGBClassifier` | Non-linear tree baseline |

XGBoost settings: `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`,
`subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `n_jobs=4`,
`random_state=42`. `BalancedXGBClassifier` sets
`scale_pos_weight = count(label 0) / count(label 1)` from the labels passed to
each fit.

## 5. Feature-treatment experiments

**Background.** 24 features contain values above 1, almost all above 10⁶ and
rounded to 3 significant digits, while their other values lie in [0, 1]. The
cause is unknown. These values are treated as unexplained observations, not as
errors (D4).

**Column list.** It is computed from training rows only: 23 columns. Over all
rows there are 24. `current_ratio` has a single value above 1 (row_id 2499),
and that row is in the test split.

| Config | Treatment | Role |
|---|---|---|
| `lr_standard` | `StandardScaler` on the original values | Candidate |
| `lr_quantile` | Quantile transform, normal output (uses only the order of values) | Candidate |
| `lr_large_as_missing` | Values > 1 → missing, median imputation, `StandardScaler` | Candidate |
| `xgb_raw` | Original values | Candidate |
| `xgb_large_as_missing` | Values > 1 → missing (XGBoost handles missing values natively) | Candidate |
| `lr_quantile_drop24` | The 23 training-derived large-value columns removed, then quantile + LR | Diagnostic only |
| `xgb_raw_drop24` | The 23 columns removed, then XGBoost | Diagnostic only |

**Rationale:**

- Tree models only use the order of values within each feature, so the
  original values can be used directly.
- Logistic Regression is scale-sensitive, so it gets a rank-based transform.
- The "large as missing" treatment tests the hypothesis that the large values
  are on a different scale.
- The drop-24 ablations measure how much signal those columns carry. They are
  not evidence that the features are defective or valid.
- No indicator columns were added.

**Selection rule (fixed before running):** per family, choose the first
treatment in preference order (LR: quantile, large-as-missing, standard;
XGBoost: raw, large-as-missing) whose mean fold-by-fold average-precision
deficit to the best treatment is at most one standard error.

| Family | Selected | Reason recorded in `selection.json` |
|---|---|---|
| Logistic Regression | `lr_quantile` | Highest mean and first in preference order; no tie-break needed |
| XGBoost | `xgb_large_as_missing` | Highest mean; `xgb_raw` was more than one SE behind (+0.0126, SE 0.0075, better in 11 of 15 folds) |

The XGBoost margin is small, and the SE understates uncertainty because
repeated-CV folds overlap (section 9).

**Influence of the large-value columns** (first CV repeat, training folds only):

| Config | Base average precision | Drop when the 23 columns are shuffled together | XGBoost gain share |
|---|---|---|---|
| `lr_quantile` | 0.3444 ± 0.0417 | 0.2761 ± 0.0501 | n/a |
| `xgb_large_as_missing` | 0.4146 ± 0.0870 | 0.1937 ± 0.0576 | 0.3065 ± 0.0565 |

Shuffling 23 columns together creates unrealistic rows, so these drops may
overstate the columns' importance.

**At test time,** `current_ratio`'s large value in the test row is handled by
each pipeline's normal rule:

- `xgb_large_as_missing` treats it as missing.
- `lr_quantile` maps it to the top of the training range, because the quantile
  transform clips values outside the range it was fitted on.

## 6. Cross-validation methodology and results

**Method:**

- Training rows only: 5,455, of which 176 have label 1.
- `RepeatedStratifiedKFold`: 5 folds × 3 repeats = 15 folds, seed 42.
- Every transformation sits inside the pipeline and is refitted per training fold.
- About 35 label-1 rows per validation fold.

**Results**, mean ± standard deviation across the 15 folds (from `cv_results.csv`):

| Config | Average precision | ROC-AUC | Brier | Log-loss | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|---|---|---|
| dummy | 0.0323 ± 0.0004 | 0.5000 ± 0.0000 | 0.0312 ± 0.0004 | 0.1425 ± 0.0013 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| lr_standard | 0.3390 ± 0.0569 | 0.8931 ± 0.0289 | 0.0925 ± 0.0083 | 0.3615 ± 0.0432 | 0.1759 ± 0.0191 | 0.7781 ± 0.0579 | 0.2867 ± 0.0279 |
| **lr_quantile** | 0.3503 ± 0.0526 | 0.9186 ± 0.0181 | 0.0843 ± 0.0075 | 0.2938 ± 0.0251 | 0.1916 ± 0.0200 | 0.7970 ± 0.0789 | 0.3085 ± 0.0295 |
| lr_large_as_missing | 0.3336 ± 0.0575 | 0.8894 ± 0.0360 | 0.0906 ± 0.0077 | 0.3662 ± 0.0579 | 0.1794 ± 0.0152 | 0.7896 ± 0.0567 | 0.2921 ± 0.0215 |
| xgb_raw | 0.4164 ± 0.0721 | 0.9321 ± 0.0171 | 0.0283 ± 0.0029 | 0.1030 ± 0.0113 | 0.4314 ± 0.0627 | 0.4847 ± 0.0770 | 0.4533 ± 0.0558 |
| **xgb_large_as_missing** | 0.4291 ± 0.0714 | 0.9343 ± 0.0148 | 0.0284 ± 0.0031 | 0.1027 ± 0.0118 | 0.4204 ± 0.0530 | 0.4713 ± 0.0654 | 0.4425 ± 0.0486 |
| lr_quantile_drop24 (diagnostic) | 0.3634 ± 0.0592 | 0.9193 ± 0.0204 | 0.0913 ± 0.0069 | 0.3101 ± 0.0235 | 0.1831 ± 0.0136 | 0.8426 ± 0.0683 | 0.3004 ± 0.0196 |
| xgb_raw_drop24 (diagnostic) | 0.3677 ± 0.0518 | 0.9233 ± 0.0194 | 0.0323 ± 0.0030 | 0.1140 ± 0.0125 | 0.3668 ± 0.0473 | 0.4429 ± 0.0608 | 0.3995 ± 0.0469 |

**Paired comparisons** (average precision, A − B, same folds, from `cv_report.md`):

- `lr_quantile` − `dummy`: +0.3180, better in 15 of 15 folds.
- `xgb_large_as_missing` − `dummy`: +0.3968, better in 15 of 15 folds.
- `xgb_large_as_missing` − `lr_quantile`: +0.0787, SE 0.0194, better in 12 of 15 folds (descriptive only).

No warnings were raised during fitting.

## 7. Held-out test results

The test set was used **once**, after the treatments and thresholds were
frozen. Each selected configuration was refitted on the full training set.
Test set: 1,364 rows, **44 with label 1**.

Results at the frozen max-F1 thresholds, with 95% stratified bootstrap intervals (2,000 resamples):

| Model | Average precision | ROC-AUC | Brier | Log-loss | Threshold | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| dummy | 0.0323 [0.032, 0.032] | 0.5000 [0.500, 0.500] | 0.0312 | 0.1425 | 0.032081 | 0.0323 | 1.0000 | 0.0625 | 0 / 1320 / 0 / 44 |
| lr_quantile | 0.4718 [0.354, 0.616] | 0.9400 [0.909, 0.964] | 0.0855 | 0.2847 | 0.886395 | 0.3857 [0.303, 0.479] | 0.6136 [0.477, 0.750] | 0.4737 [0.374, 0.569] | 1277 / 43 / 17 / 27 |
| xgb_large_as_missing | 0.4871 [0.364, 0.637] | 0.9579 [0.938, 0.974] | 0.0271 | 0.0924 | 0.398406 | 0.4179 [0.325, 0.517] | 0.6364 [0.500, 0.773] | 0.5045 [0.396, 0.602] | 1281 / 39 / 16 / 28 |

Reference, at threshold 0.5:

| Model | Precision | Recall | F1 | TN / FP / FN / TP |
|---|---|---|---|---|
| dummy | 0.0000 | 0.0000 | 0.0000 | 1320 / 0 / 44 / 0 |
| lr_quantile | 0.1917 | 0.8409 | 0.3122 | 1164 / 156 / 7 / 37 |
| xgb_large_as_missing | 0.4483 | 0.5909 | 0.5098 | 1288 / 32 / 18 / 26 |

Paired bootstrap differences in average precision:

| Comparison | Mean difference | 95% interval |
|---|---|---|
| lr_quantile − dummy | +0.4482 | [+0.3215, +0.5842] |
| xgb_large_as_missing − dummy | +0.4670 | [+0.3318, +0.6043] |
| xgb_large_as_missing − lr_quantile | +0.0188 | [−0.1208, +0.1629] |

**Interpretation:**

- Both models rank label-1 rows far better than the Dummy.
- The test set does not distinguish Logistic Regression from XGBoost: the
  interval for their difference includes zero. **No winner is chosen.**
- The test scores are higher than the CV means (LR 0.350 → 0.472, XGBoost
  0.429 → 0.487). Possible contributors, which were not tested, are sampling
  variation with 44 label-1 rows and the larger training set for the final fit
  (5,455 rows versus about 4,364 per CV fold).
- The XGBoost CV mean lies inside its test interval. The LR CV mean (0.350)
  lies just below its test interval (0.354), which is consistent with a
  favourable test sample but does not prove one.

## 8. Thresholds

| Model | Frozen threshold | Selection |
|---|---|---|
| dummy | 0.032081 | Max F1 on pooled training out-of-fold predictions (flags every row) |
| lr_quantile | 0.886395 | Max F1 on pooled training out-of-fold predictions |
| xgb_large_as_missing | 0.398406 | Max F1 on pooled training out-of-fold predictions |

- The thresholds were chosen using **training out-of-fold predictions only**,
  pooled across the 3 repeats. No test label was used.
- Threshold 0.5 is reported as a reference.
- The out-of-fold precision, recall and F1 at these thresholds (in
  `cv_report.md`) are optimistic, because the threshold was chosen on those
  same predictions.
- F1 weights false positives and false negatives equally. Because the label
  meaning is unverified, the real cost of each error type is unknown.

## 9. Uncertainty

- **Test intervals** are percentile intervals from a stratified bootstrap of
  the test set (2,000 resamples, seed 42). They reflect test-set sampling only,
  not variation from the choice of split or from training. With 44 label-1
  rows they are approximate.
- **One label-1 test row** changes recall by 2.3 percentage points.
- **CV spread** (± values) is the standard deviation of fold scores. Repeated
  CV reuses the same rows, so fold scores are correlated. The spread is not a
  confidence interval, and paired-comparison standard errors understate
  uncertainty.
- **Probabilities are not calibrated.** The predicted scores run high, which
  is consistent with the class weighting. No unweighted model was run to
  confirm the cause.
  - On pooled CV predictions, `lr_quantile` rows predicted at 0.9–1.0 had an
    observed label-1 share of 0.36, and `xgb_large_as_missing` rows at 0.9–1.0
    had 0.64.
  - LR's test Brier score (0.0855) is worse than the Dummy's (0.0312).
  - Test-set calibration tables were not computed.
  - Treat the outputs as ranking scores, not probabilities.

## 10. Leakage safeguards and reproducibility

**Leakage safeguards:**

- **Two stages.** `--stage cv` requests training rows only from the loader, so
  test rows never reach the CV code. `--stage final` requires a frozen
  `selection.json` and refuses to overwrite results without `--force`.
- **Pipelines.** Every data-dependent step (quantile transform, imputation,
  scaling) is inside a pipeline and refitted per training fold.
- **Class weights** come from each fit's labels.
- **Thresholds and treatments** are chosen from training CV only.
- **Large-value column list** is computed from training-row feature values
  only, with no labels.
- **Tests:** a unit test on synthetic data checks that every CV fit sees only
  its fold's training rows and no test row. A second test runs the CV stage on
  the real training rows and checks that the out-of-fold predictions cover
  exactly the 5,455 training rows. The full two-stage test runs on synthetic
  data only, so no test evaluates the real test set.
- **Checksums:** each stage verifies the processed files, and the raw CSV,
  before reading them and again before writing any output. If anything
  changed, it writes nothing. The final stage also checks that the processed
  files match those used in the CV stage. The raw CSV is verified only when
  the default `--input-dir` is used. Independent `sha256sum` checks after both
  stages found all files unchanged.

**Reproducibility:**

- **Seeds:** 42 for the split, CV, XGBoost, the quantile transform, the
  bootstrap, and permutation.
- **Threads:** XGBoost `n_jobs=4`.
- **Versions:** Python 3.12.10, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1,
  xgboost 3.4.1, joblib 1.6.0 (full environment in `requirements-lock.txt`).
- **Recorded in metadata:** the git commit, the settings snapshot, and the
  input checksums (`selection.json`, `test_metrics.json`, `model_metadata.json`).
- **Verified rerun:** re-running the CV stage reproduced `cv_results.csv` and
  `oof_predictions.csv` byte for byte. Reproducibility on other machines or
  thread counts has not been verified.

## 11. Limitations, ethical considerations, and out-of-scope uses

**Limitations:**

- Label meanings are unverified.
- Small minority class: 176 training and 44 test label-1 rows.
- No company identifier: the same company may appear in both train and test,
  which could make results optimistic.
- No date column: no time-ordered evaluation, and no evidence of performance
  on future data.
- Historical (1999–2009), Taiwan-only data.
- Undocumented feature units and definitions. Unexplained large values in 24
  features.
- Uncalibrated probabilities.
- Fixed settings with no tuning. Results could change with other settings.

**Ethical considerations:**

- A score attached to a named company could harm it, for example through
  credit or reputational damage. This is especially serious because what the
  label represents is unconfirmed.
- False positives and false negatives have real-world costs that this project
  has not assessed.
- The data covers a specific market and period. Applying the models elsewhere
  could systematically misjudge companies.

**Out of scope:**

- Lending, investment, credit-rating, insurance, audit, or regulatory decisions.
- Any statement that a specific company is or will become bankrupt.
- Use on companies outside Taiwan, outside 1999–2009, or with features
  computed differently.
- Use of the scores as calibrated probabilities.
- Financial advice of any kind.

## 12. Artifacts and reproduction

**Model artifacts** (ignored by Git), in `artifacts/models/taiwan_bankruptcy/`:

| File | SHA-256 |
|---|---|
| `dummy.joblib` | `ecb7d7e07de8cb7dde38aa58e62015dc8111a795950857f12706aca2a5a68afc` |
| `lr_quantile.joblib` | `37744485751beaf9cba8fd1579ec5c06b9aa52f7299a62a9e00d4c9b712641ba` |
| `xgb_large_as_missing.joblib` | `7a7c477e9598f8d1dc64402375f056048e252ee28ce599d400e705fd07817939` |
| `model_metadata.json` | settings, seed, git commit, versions |

Each model is fitted on the full training split. joblib files can run code
when loaded, so load only files created by this project.

**Reports** (ignored by Git), in `data/reports/taiwan_bankruptcy/baselines/`:
`cv_report.md`, `cv_results.csv`, `oof_predictions.csv`,
`permutation_importance.csv`, `selection.json`, `final_report.md`,
`test_metrics.json`, `test_predictions.csv`.

**Reproduction**, from the project root with the project's virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\download_taiwan_bankruptcy.py
.\.venv\Scripts\python.exe scripts\preprocess_taiwan_bankruptcy.py
.\.venv\Scripts\python.exe scripts\train_distress_baselines.py --stage cv
.\.venv\Scripts\python.exe scripts\train_distress_baselines.py --stage final
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

- Every script refuses to overwrite existing outputs without `--force`. The
  downloader skips an existing, checksum-verified download.
- To reproduce the evaluation without touching existing results, give
  `train_distress_baselines.py` new `--output-dir` and `--artifacts-dir`
  folders. It keeps reading the processed data in `data/interim/`.
- If you also write preprocessing to a new folder (`--output-dir` on the
  preprocessing script), pass that folder as `--input-dir`. The raw-CSV check
  is then skipped, as described in section 10.
- A fresh download is checked against the recorded checksum only on reruns.
  Compare `data/raw/taiwan_bankruptcy/metadata.json` with the raw CSV SHA-256
  in section 2. UCI may change the file.
