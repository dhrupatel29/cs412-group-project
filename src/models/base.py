"""Shared model interface. Every model in this project follows it so experiments/run.py
can treat them the same.

    model = SomeModel(num_cols, cat_cols, seed=42, **config)
    model.fit(X_train, y_train, X_val, y_val)   # raw DataFrames; NaN allowed
    p = model.predict_proba(X_test)              # 1-D array of P(y=1)

Rules:
- Fit any preprocessing on X_train only.
- X_val may be used for early stopping / tuning, never for fitting preprocessing.
- Inputs at test time may contain NaN (masked cells) in any column.
"""


class BaseModel:
    def __init__(self, num_cols, cat_cols, seed=42, **config):
        self.num_cols, self.cat_cols, self.seed, self.config = num_cols, cat_cols, seed, config

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError
