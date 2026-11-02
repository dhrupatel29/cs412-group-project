"""FT-Transformer (Gorishniy et al., NeurIPS 2021). OWNER: Pranay.

TODO
- Follow the interface in src/models/base.py.
- Numerical features -> per-feature linear embeddings; categorical -> per-feature
  embedding tables; prepend [CLS]; Transformer encoder; predict from [CLS].
  The `rtdl_revisiting_models` package has a reference implementation; using it is fine
  if you cite it, writing your own is also fine.
- Missing values at test time: decide and document how masked cells are handled
  (e.g. median/"__missing__" imputation like the baselines, or a learned [MASK] embedding).
  The baseline route is the fair default; anything else is an extra condition.
- Categorical vocab: build from train only; unseen/missing -> a reserved index.
- Early stopping on X_val; seed everything with self.seed.
- Record the config you settle on in configs/ft_transformer.yaml.
"""
from src.models.base import BaseModel


class FTTransformer(BaseModel):
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError("FT-Transformer not implemented yet (owner: Pranay)")

    def predict_proba(self, X):
        raise NotImplementedError
