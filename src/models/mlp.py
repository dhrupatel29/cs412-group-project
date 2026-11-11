"""MLP baseline. OWNER: Vaibhav.

Input: the same representation as LR/XGB (src.data.make_preprocessor, fit on train only):
median-imputed + standardized numerics, one-hot categoricals. Masked numeric cells get the
train median; masked categorical cells become '__missing__', which is unseen in train and
therefore encodes as an all-zero one-hot block (handle_unknown='ignore').

Network: [Linear -> ReLU -> Dropout] x n_layers -> Linear(1 logit).
Training: AdamW + BCEWithLogitsLoss, minibatches, early stopping on validation AP,
best-epoch weights restored. No class weighting or resampling.
"""
import copy
import random

import numpy as np
from sklearn.metrics import average_precision_score

from src.data import make_preprocessor
from src.models.base import BaseModel

DEFAULTS = dict(
    n_layers=2, width=256, dropout=0.1,
    lr=1e-3, weight_decay=1e-5,
    batch_size=256, max_epochs=100, patience=10,
    # 1 thread: on macOS, multi-threaded torch segfaults when sklearn/xgboost load their
    # own OpenMP runtime first (as experiments/run.py does). A small MLP barely needs more.
    num_threads=1, verbose=False,
)


class MLP(BaseModel):
    def __init__(self, num_cols, cat_cols, seed=42, **config):
        super().__init__(list(num_cols), list(cat_cols), seed, **config)
        unknown = set(config) - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unsupported MLP options: {sorted(unknown)}")
        self.settings = {**DEFAULTS, **config}
        for key in ["n_layers", "width", "batch_size", "max_epochs", "patience", "num_threads"]:
            if not isinstance(self.settings[key], int) or self.settings[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 <= self.settings["dropout"] < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.settings["lr"] <= 0 or self.settings["weight_decay"] < 0:
            raise ValueError("lr must be positive and weight_decay non-negative")

    def _build(self, n_in):
        import torch.nn as nn

        cfg, layers, d = self.settings, [], n_in
        for _ in range(cfg["n_layers"]):
            layers += [nn.Linear(d, cfg["width"]), nn.ReLU(), nn.Dropout(cfg["dropout"])]
            d = cfg["width"]
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def _tensor(self, X):
        import torch

        return torch.as_tensor(self.prep_.transform(X), dtype=torch.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch

        if X_val is None or y_val is None or len(X_val) == 0:
            raise ValueError("Validation data is required for AP early stopping")
        cfg = self.settings
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(cfg["num_threads"])

        self.prep_ = make_preprocessor(self.num_cols, self.cat_cols).fit(X_train)
        xt, xv = self._tensor(X_train), self._tensor(X_val)
        yt = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
        yv = np.asarray(y_val)

        self.model_ = self._build(xt.shape[1])
        opt = torch.optim.AdamW(self.model_.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(self.seed)

        self.history_ = []
        self.best_val_ap_, self.best_epoch_ = -np.inf, 0
        best_state, stale = None, 0
        for epoch in range(1, cfg["max_epochs"] + 1):
            self.model_.train()
            total = 0.0
            for ix in torch.randperm(len(yt), generator=gen).split(cfg["batch_size"]):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xt[ix]).squeeze(-1), yt[ix])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(ix)

            ap = float(average_precision_score(yv, self._predict_tensor(xv)))
            self.history_.append(dict(epoch=epoch, train_loss=total / len(yt), val_ap=ap))
            if ap > self.best_val_ap_:
                self.best_val_ap_, self.best_epoch_ = ap, epoch
                best_state, stale = copy.deepcopy(self.model_.state_dict()), 0
            else:
                stale += 1
            if cfg["verbose"]:
                print(f"MLP epoch {epoch}: val AP={ap:.4f}, best={self.best_val_ap_:.4f}", flush=True)
            if stale >= cfg["patience"]:
                break

        self.epochs_trained_ = len(self.history_)
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def _predict_tensor(self, x):
        import torch

        self.model_.eval()
        with torch.inference_mode():
            return torch.sigmoid(self.model_(x).squeeze(-1)).numpy()

    def predict_proba(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("Call fit before predict_proba")
        return self._predict_tensor(self._tensor(X))
