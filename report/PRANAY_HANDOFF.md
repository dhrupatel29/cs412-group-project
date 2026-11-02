# Pranay: implementation and handoff

## Dates from the shared tracker

- Nov 3: setup and pilot check. Already completed by Pranay before this work.
- Nov 8: FT-Transformer end-to-end and missing-input handling. Implementation and
  focused checks are in `src/models/ft_transformer.py` and `tests/`. Completed
  Nov 2: both full-data splits and all four masking conditions ran successfully
  in a two-epoch integration check; all eight metric rows are in
  `results/pranay/smoke/ftt_metrics.csv`. CPU: Intel Core Ultra 7 155U, four torch
  threads, CPU-only PyTorch 2.11.0. These are development checks, not final results.
- Nov 15: finish 20 chronological validation-only trials and commit the selected
  `configs/ft_transformer.yaml`. Runtime summary code is ready now.
- Nov 24: produce the final four-model runtime table after final runs are available.
- Dec 1: integrate Pranay's report sections and slides into the shared deliverables.

The tracker schedules a team protocol decision for Nov 4. The supplied guide
specifies split seed 42, 20 trials, validation AP, and the common imputation
condition. Confirm those shared decisions before treating a configuration as
frozen. The development YAML is deliberately named `ft_transformer_initial.yaml`.

Pranay chose to finish the Nov 8 milestone now and leave the full tuning search
for the Nov 9-15 period. No 20-trial search or frozen tuned config is claimed.

## Commands (PowerShell, repository root)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Quick full-data integration check, not a tuned or final result:
.\.venv\Scripts\python.exe -m experiments.ftt evaluate --config configs/ft_transformer_initial.yaml --max-epochs 2 --patience 1 --out-dir results/pranay/smoke
# Nov 9-15: run after the team locks the tuning protocol (CPU can take hours):
.\.venv\Scripts\python.exe -m experiments.ftt tune --trials 20
# Inspect all 20 rows in results/pranay/ftt_tuning.csv before freezing.
.\.venv\Scripts\python.exe -m experiments.ftt evaluate --config configs/ft_transformer.yaml --out-dir results/pranay/tuned_check
# Explicitly select the latest appended pilot rows; do not mix repeated runs:
.\.venv\Scripts\python.exe -m experiments.runtime --input results/metrics.csv --latest-per-condition --out results/pranay/runtime_pilot.csv
```

The evaluator/tuner uses Pranay's own script so it does not change Dhru's
`experiments/run.py`, overwrite the shared pilot CSV, or overwrite existing
baseline predictions. Its small integration run uses two epochs and one seed,
and must not be presented as the final tuned comparison. The common runner can
also use the implemented `ftt` registry entry with its class defaults. Dhru's
upcoming config loader should pass the selected flat YAML dictionary to that
constructor (not silently ignore it).

Tuning reads chronological train and validation slices only. It records every
trial and its validation AP. Only a completed 20-trial run writes the final config
filename. An interrupted run has no final frozen config; inspect its saved
best-so-far file and log before rerunning.

## Missing inputs

Numerical values use training medians and standardization. Categorical vocabularies
use only observed training values; missing/unseen inputs map to index 0 with a fixed
zero embedding. Other categories, including the literal `unknown`, have their own
trainable embeddings. No training mask augmentation or learned mask-token variant
is included. All preprocessing stays inside `fit`, and prediction does not extend
the vocabulary.

## Final dependencies

Vaibhav's MLP is still a placeholder at this checkpoint. Dhru owns the final
five-seed experiment runs and shared config loader. The runtime table cannot be
called a final four-model comparison until those inputs exist. Run aggregation
on clean rows only, record hardware from the actual measuring process, and do not
claim that existing pilot timings have a verified common hardware protocol.

`pranay_sections.tex` and `pranay_references.bib` contain Pranay's report sections.
`pranay_slides.md` contains his three-slide content and timed speaking notes.
These need integration into the shared report/deck and a final results check.
The remaining team members' files were not edited.

Prepared with AI assistance for Pranay to review and understand before presentation.
