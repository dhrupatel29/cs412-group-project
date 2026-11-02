# Task split

Rule: everyone commits their own work from their own GitHub account. The course grades contributions from the commit history.

## Dhru: pipeline, experiments, results
- [ ] Create the GitHub repo, add Pranay + Vaibhav as collaborators
- [ ] Download data and reproduce the pilot (`python -m experiments.run --models logreg xgb --seeds 42`). Check that it matches Table 1 of the intermediate report.
- [ ] Own `src/data.py`, `src/masking.py`, `src/metrics.py`, `experiments/run.py`
- [ ] Run the final 5-seed clean + missingness experiments for all 4 models
- [ ] Aggregate results (mean ± std over seeds), build the results tables and missingness plots
- [ ] Runtime: fit time + inference latency table
- [ ] Report: Results section

## Pranay: FT-Transformer
- [ ] Implement `src/models/ft_transformer.py` (see TODO in the file)
- [ ] Decide and document how FT-Transformer handles masked inputs
- [ ] Tune FT-Transformer on validation with a fixed, recorded budget; save to `configs/ft_transformer.yaml`
- [ ] Report: Methods section on FT-Transformer and related work (intro and survey)

## Vaibhav: MLP, tuning, shift and calibration analysis
- [ ] Implement `src/models/mlp.py`
- [ ] Tune MLP, LR, and XGB on validation with the same kind of recorded budget; save configs to `configs/`
- [ ] Reliability plots (calibration) per model per split
- [ ] Shift analysis: train vs. test class prevalence and feature distribution shift (e.g. `euribor3m`, `nr.employed`, month)
- [ ] Report: Data section, plus the calibration/shift discussion

## Shared
- [ ] Final report (8–10 pages, ACM SIG or CVPR LaTeX) in `report/`
- [ ] 5–8 min presentation: each person presents their part
- [ ] Contributions section listing who did what

## Timeline
| Week | Goal |
|---|---|
| Nov 2–8 | Repo up, pilot reproduced, MLP + FT-Transformer running end-to-end |
| Nov 9–15 | Tuning done, configs frozen |
| Nov 16–22 | 5-seed clean + missingness runs, calibration + shift analysis |
| Nov 23–Dec 1 | Runtime benchmark, figures, report, slides |

## Decisions to lock this week
- Random split: fixed across seeds (default `--split-seed 42`), with only model seeds varying. Change this only if everyone agrees.
- Tuning budget: same number of trials per model (e.g. 20 random-search trials), selected on validation AP.
- Missing-value handling: by default every model uses the same imputation as the baselines. Any learned-mask variant is an extra condition, not a replacement.
