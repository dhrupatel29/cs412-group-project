# Pranay's section of the team presentation

Approximately 2 minutes within the shared 5-8 minute presentation.
Slide text and notes are ready for the team's chosen slide template. Add final
findings only after Dhru's frozen five-seed runs and Vaibhav's analysis exist.

## Slide 1: The prediction and evaluation problem

- Predict whether a customer subscribes to a bank term deposit
- Exclude call duration because the prediction precedes the call outcome
- Compare random and chronological evaluation
- Stress-test the same trained models with 10%, 20%, and 30% missing input cells

Speaker notes (35 seconds):
Our question is whether a model that works on a random split also works on later
records. Those test populations differ, so we inspect prevalence and feature drift
instead of treating the split comparison as a causal experiment. We also use the
same missing-value masks across models to compare their response to incomplete
inputs. Hand the detailed dataset discussion to Vaibhav.

## Slide 2: FT-Transformer

- One embedding token per numerical or categorical feature
- Attention learns interactions across columns
- The final CLS representation produces a subscription probability
- Validation average precision selects the best training epoch

Speaker notes (45 seconds):
We adapted the authors' official FT-Transformer implementation to the team's
model interface. Numerical values use training medians and standardization.
Categorical vocabularies also come only from training data. Missing and unseen
categories share a reserved zero embedding; the dataset's literal unknown string
remains an ordinary category. We restore the weights from the best validation-AP
epoch before predicting. This is the base imputation condition, without a learned
mask-token variant.

## Slide 3: What makes the comparison credible

- Training-only preprocessing and validation-only model selection
- Fixed split seed, with model seeds varied in the final runs
- Identical missing-input masks for every model
- Runtime summaries count each fitted model once per seed

Speaker notes (40 seconds):
Each masking level reuses a fitted model. We therefore do not count its repeated
fit-time values as separate training runs. The final runtime table needs all four
models on the same machine with documented hardware and timing conditions. Our
development checks establish that the adapter handles missing inputs and repeats
the same predictions for the same seed. Final accuracy and runtime conclusions
will come from the shared evaluation. Hand the results to Dhru.

Sources for the team's notes:
- Gorishniy et al., NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html
- Official implementation: https://github.com/yandex-research/rtdl-revisiting-models
- UCI Bank Marketing: https://doi.org/10.24432/C5K306
