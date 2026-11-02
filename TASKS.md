# Task split

Rule: everyone commits their own work from their own GitHub account. Course grades contributions from the commit history.

Each person owns one model-building job, one analysis job, and one report section.

## Dhru: pipeline, baselines, final runs
- [ ] Create GitHub repo, add Pranay + Vaibhav as collaborators
- [ ] Download data, reproduce pilot (`python -m experiments.run --models logreg xgb --seeds 42`), compare to Table 1 of intermediate report
- [ ] Update `experiments/run.py` to save predicted probabilities to `results/` (Vaibhav needs these for calibration plots)
- [ ] Maintain `src/data.py`, `src/masking.py`, `src/metrics.py`, `experiments/run.py`
- [ ] Tune LR + XGB with the shared budget; save to `configs/`
- [ ] Run final 5-seed clean + missingness experiments for all 4 models
- [ ] Report: Results section

## Pranay: FT-Transformer + runtime
- [ ] Implement `src/models/ft_transformer.py` (see TODO in file)
- [ ] Decide and document how FT-Transformer handles masked inputs
- [ ] Tune FT-Transformer with the shared budget; save to `configs/ft_transformer.yaml`
- [ ] Runtime: fit time + inference latency table for all 4 models
- [ ] Report: Introduction, related work, FT-Transformer method section

## Vaibhav: MLP + analysis
- [ ] Implement `src/models/mlp.py`
- [ ] Tune MLP with the shared budget; save to `configs/mlp.yaml`
- [ ] Aggregate results (mean ± std over seeds) into tables, plus missingness plots
- [ ] Reliability plots (calibration) per model per split
- [ ] Shift analysis: train vs. test class prevalence and feature drift (e.g. `euribor3m`, `nr.employed`, month)
- [ ] Report: Data section, plus calibration/shift discussion

## Shared
- [ ] Conclusion + future work
- [ ] 5–8 min presentation (each person presents their part)
- [ ] Contributions section listing who did what
- [ ] Final report (8–10 pages, ACM SIG or CVPR LaTeX) in `report/`

## What can run in parallel
Start now, no waiting:
- Dhru: reproduce pilot, add prediction saving, tune LR + XGB
- Pranay: build FT-Transformer, write intro + related work
- Vaibhav: build MLP, shift analysis, write Data section

Waiting points:
1. Final 5-seed runs (Dhru) → need MLP + FT-Transformer working and all configs tuned
2. Results tables + plots (Vaibhav) → need final runs. Write scripts now against pilot `results/metrics.csv`, re-run later
3. Runtime table (Pranay) → needs all 4 models. Build on LR + XGB now, add the rest later
4. Results section (Dhru) → after final runs

## Timeline
| Week | Goal |
|---|---|
| Nov 2–8 | Repo up, pilot reproduced, MLP + FT-Transformer running end-to-end |
| Nov 9–15 | Tuning done, configs frozen |
| Nov 16–22 | 5-seed clean + missingness runs, calibration + shift analysis |
| Nov 23–Dec 1 | Runtime table, figures, report, slides |

## Decisions to lock this week
- [ ] Random split fixed across seeds (`--split-seed 42`); only model seeds vary
- [ ] Tuning budget: same for every model (e.g. 20 random-search trials), selected on validation AP
- [ ] Missing-value handling: every model uses the baseline imputation by default; any learned-mask variant is an extra condition, not a replacement





# Guide for Pranay & Vaibhav

## Setup (both, do this first)
1. Accept the GitHub invite from dhrupatel29
2. Clone and install:
   ```
   git clone https://github.com/dhrupatel29/cs412-group-project.git
   cd cs412-group-project
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download data: https://archive.ics.uci.edu/dataset/222/bank+marketing
   Unzip, open the inner `bank-additional.zip`, put `bank-additional-full.csv` in `data/` (it's gitignored, never commit it)
4. Sanity check: `python -m experiments.run --models logreg xgb --seeds 42` should run and print metrics
5. Read `src/models/base.py` (the interface) and `src/models/baselines.py` (a working example)

## Git workflow
- You each mostly edit your own files, so working on `main` is fine
- Always `git pull` before starting, commit small and often, `git push` at the end
- Commit from YOUR account; the grade comes from the commit history
- Don't edit `src/data.py`, `src/masking.py`, `src/metrics.py`, `experiments/run.py` without telling Dhru

## Rules every model follows
- Class takes `(num_cols, cat_cols, seed=42, **config)`
- `fit(X_train, y_train, X_val, y_val)` gets raw pandas DataFrames
- `predict_proba(X)` returns a 1-D numpy array of P(y=1)
- Fit any preprocessing on train only. Val is for early stopping / tuning only
- Test inputs can contain NaN anywhere (masked cells), so the model must not crash on them
- No class weighting / resampling (the baselines don't use any, so the comparison stays fair)
- Seed everything with `self.seed`: `torch.manual_seed`, `np.random.seed`, `random.seed`
- Selection metric for early stopping and tuning: validation AP (`sklearn.metrics.average_precision_score`)

## Tuning budget (same for all models)
- 20 random-search trials per model, on the **chronological** split, seed 42
- Pick the config with the best validation AP
- Save it to `configs/<model>.yaml` and note the search ranges in a comment at the top
- Don't look at test metrics while tuning

---

## Pranay: FT-Transformer + runtime

### 1. Build FT-Transformer (`src/models/ft_transformer.py`), by Nov 8
- Simplest route: `pip install rtdl_revisiting_models` (Gorishniy et al.'s official package), add it to `requirements.txt`, cite it in the report
- Preprocessing inside `fit` (train only):
  - Numerical: median impute + standardize (`SimpleImputer` + `StandardScaler`)
  - Categorical: build a vocab per column from train; map each value to an int. Reserve one index for missing/unseen, so cardinality = len(vocab) + 1. The dataset's own `"unknown"` stays a normal category
- Model: `FTTransformer(n_cont_features=..., cat_cardinalities=[...], d_out=1, **FTTransformer.get_default_kwargs())`
- Training: AdamW (lr 1e-4, weight decay 1e-5), batch 256, `BCEWithLogitsLoss`, up to 100 epochs, early stopping on val AP with patience ~16, restore best weights
- `predict_proba`: apply the same preprocessing, then sigmoid of the output
- Masked inputs: default is the same as the baselines (median for numbers, reserved index for categories). Write 2–3 sentences in the report on this choice. A learned [MASK] token is optional, and only as an extra condition
- Test it: `python -m experiments.run --models ftt --seeds 42`

### 2. Tune it, Nov 9–15
- Search: `n_blocks` 1–4, `d_block` {96, 128, 192, 256}, `attention_dropout` 0.1–0.4, `ffn_dropout` 0–0.3, lr 3e-5–3e-4, weight decay 1e-6–1e-4
- 20 trials, best val AP → `configs/ft_transformer.yaml`

### 3. Runtime table
- `results/metrics.csv` already has `fit_s` and `infer_ms_per_row`
- Write `experiments/runtime.py`: average these over seeds per model and split, and output a table
- State hardware in the report (CPU model, GPU or not). Run all models on the same machine
- Build it now on LR + XGB, re-run once MLP + FT-Transformer exist

### 4. Report
- Introduction (problem, why temporal shift + missingness matter, summary of results)
- Related work (XGBoost, FT-Transformer, tabular DL vs trees; reuse the proposal survey)
- Method section on FT-Transformer (tokenization, attention, CLS head, how masking is handled)

---

## Vaibhav: MLP + analysis

### 1. Build the MLP (`src/models/mlp.py`), by Nov 8
- Preprocessing: reuse `make_preprocessor(self.num_cols, self.cat_cols)` from `src/data.py`, fit on train, output is a dense one-hot + scaled matrix (same input as LR/XGB)
- Model (PyTorch): 2–3 hidden layers of 256, ReLU, dropout 0.1, output 1 logit
- Training: AdamW (lr 1e-3, weight decay 1e-5), batch 256, `BCEWithLogitsLoss`, up to 100 epochs, early stopping on val AP with patience ~10, restore best weights
- `predict_proba`: sigmoid of the output
- Test it: `python -m experiments.run --models mlp --seeds 42`

### 2. Tune it, Nov 9–15
- Search: layers 1–4, width {64, 128, 256, 512}, dropout 0–0.5, lr 1e-4–3e-3, weight decay 1e-6–1e-3
- 20 trials, best val AP → `configs/mlp.yaml`

### 3. Shift analysis (start now, only needs data)
- Write `experiments/shift.py` using `load()` + `split(..., "chronological")`
- Positive rate per partition (train 5.57%, val 10.65%, test 38.45%)
- Numerical drift: train vs test mean + KS statistic (`scipy.stats.ks_2samp`) per numerical column; focus on `euribor3m`, `nr.employed`, `emp.var.rate`, `cons.conf.idx`
- Categorical drift: train vs test proportions for `month`, `poutcome`, `contact`
- One plot: `euribor3m` vs row index with vertical lines at the train/val/test boundaries
- Save figures to `results/figures/`

### 4. Results tables + missingness plots
- Write `experiments/summarize.py`: read `results/metrics.csv`, group by split / model / mask_rate, output mean ± std for AUROC, AP, F1, Brier
- Plot: AP vs mask rate, one line per model, error bars = std over seeds, one panel per split
- Build it now on the pilot CSV, re-run after the final 5-seed runs

### 5. Calibration (reliability) plots
- Dhru is adding prediction saving to the runner: files `results/preds/{split}_{model}_seed{seed}_mask{rate}.csv` with columns `y`, `p`
- Use `sklearn.calibration.calibration_curve(y, p, n_bins=10)` on clean test predictions
- One plot per split, one line per model, diagonal = perfect calibration
- Note: chronological test prevalence (38%) is very different from train (6%), so expect underprediction. Explain that it's a prevalence shift, not only a bad model

### 6. Report
- Data section (dataset, features, duration exclusion, splits table, class imbalance)
- Calibration + shift discussion inside Results

---

## Deadlines
| Date | Pranay | Vaibhav |
|---|---|---|
| Nov 8 | FT-Transformer runs end-to-end | MLP runs end-to-end, shift analysis done |
| Nov 15 | Tuned config committed | Tuned config committed, summarize script ready |
| Nov 22 | Runtime table | Tables, missingness plots, calibration plots |
| Dec 1 | Report sections + slides | Report sections + slides |

