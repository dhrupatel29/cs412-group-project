# Testing FT-Transformer Under Temporal Shift

CS 412 (UIC) course project by Pranay Karakoti, Dhru Patel, and Vaibhav Bharati.

The project compares Logistic Regression, MLP, XGBoost, and FT-Transformer on the UCI Bank Marketing dataset (term-deposit subscription). It looks at two things:
1. **Random vs. chronological** 70/15/15 splits.
2. **Test-time missingness**: 10/20/30% of input cells masked.

Call `duration` is excluded because it leaks the outcome.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put bank-additional-full.csv in data/ (see data/README.md)
```

## Run
```bash
python -m experiments.run --models logreg xgb --seeds 42               # pilot
python -m experiments.run --models logreg xgb mlp ftt --seeds 0 1 2 3 4  # full
```
Results are appended to `results/metrics.csv`.

## Layout
```
src/data.py              load, split, preprocess (fit on train only)
src/masking.py           nested test-time masks (seed 2026)
src/metrics.py           AUROC, AP, F1 @ val-F1 threshold, Brier
src/models/base.py       shared model interface
src/models/baselines.py  LogReg, XGB
src/models/mlp.py        MLP (Vaibhav)
src/models/ft_transformer.py  FT-Transformer (Pranay)
experiments/run.py       experiment runner
configs/                 tuned configs
report/                  LaTeX report
```

See `TASKS.md` for who owns what.
