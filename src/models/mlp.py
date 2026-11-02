"""MLP baseline. OWNER: Vaibhav.

TODO
- Follow the interface in src/models/base.py.
- Reuse src.data.make_preprocessor (same one-hot representation as LR/XGB) or justify a change.
- PyTorch is suggested. Use X_val for early stopping on validation loss.
- Seed torch/numpy with self.seed so the 5-seed runs differ only by seed.
- Record the config you settle on in configs/mlp.yaml.
"""
from src.models.base import BaseModel


class MLP(BaseModel):
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError("MLP not implemented yet (owner: Vaibhav)")

    def predict_proba(self, X):
        raise NotImplementedError
