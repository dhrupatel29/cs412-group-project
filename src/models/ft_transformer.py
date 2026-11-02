"""FT-Transformer (Gorishniy et al., NeurIPS 2021). OWNER: Pranay.

The official architecture, exposed through the project's raw-data model interface.
Tuned parameters live separately from these initial defaults.
"""
import copy
import random

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class FTTransformer(BaseModel):
    """Adapter for rtdl_revisiting_models 0.0.2, trained with validation AP.

    Train-only numerical medians/scaling; train-only categorical vocabularies.
    Index 0 is a fixed zero embedding for missing/unseen categories. The original
    'unknown' value remains a normal category. No learned mask-token condition.
    Reference: https://github.com/yandex-research/rtdl-revisiting-models
    """

    def __init__(self, num_cols, cat_cols, seed=42, **config):
        super().__init__(list(num_cols), list(cat_cols), seed, **config)
        self.settings = dict(n_blocks=3, d_block=192, attention_dropout=.2,
                             ffn_dropout=.1, lr=1e-4, weight_decay=1e-5,
                             batch_size=256, max_epochs=100, patience=16,
                             device='cpu', num_threads=4, verbose=False)
        unknown = set(config) - self.settings.keys()
        if unknown:
            raise ValueError(f'Unsupported FT-Transformer options: {sorted(unknown)}')
        self.settings.update(config)
        for key in ['n_blocks', 'd_block', 'batch_size', 'max_epochs', 'patience', 'num_threads']:
            if not isinstance(self.settings[key], int) or self.settings[key] < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.settings['d_block'] % 8:
            raise ValueError('d_block must be divisible by eight attention heads')
        for key in ['attention_dropout', 'ffn_dropout']:
            if not 0 <= self.settings[key] < 1:
                raise ValueError(f'{key} must be in [0, 1)')
        if self.settings['lr'] <= 0 or self.settings['weight_decay'] < 0:
            raise ValueError('lr must be positive and weight_decay non-negative')

    def _fit_preprocessing(self, X):
        if self.num_cols:
            self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
            self.scaler_ = StandardScaler().fit(self.imputer_.fit_transform(X[self.num_cols]))
        self.vocabs_ = {col: {v: i + 1 for i, v in enumerate(pd.unique(X[col].dropna()))}
                        for col in self.cat_cols}

    def _transform(self, X):
        xn = xc = None
        if self.num_cols:
            xn = self.scaler_.transform(self.imputer_.transform(X[self.num_cols])).astype(np.float32)
        if self.cat_cols:
            xc = np.column_stack([X[c].astype(object).map(self.vocabs_[c]).fillna(0).to_numpy(dtype=np.int64)
                                  for c in self.cat_cols])
        return xn, xc

    def _tensors(self, X):
        import torch
        return tuple(None if a is None else torch.as_tensor(a, device=self.device_)
                     for a in self._transform(X))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        from rtdl_revisiting_models import FTTransformer as ReferenceFTTransformer

        if X_val is None or y_val is None or len(X_val) == 0:
            raise ValueError('Separate non-empty validation data is required for AP early stopping')
        y, yv = np.asarray(y_train), np.asarray(y_val)
        if y.shape != (len(X_train),) or yv.shape != (len(X_val),):
            raise ValueError('Labels must be 1-D and match the input rows')
        if set(np.unique(y)) != {0, 1} or set(np.unique(yv)) != {0, 1}:
            raise ValueError('Train and validation labels must each contain both binary classes')
        cfg = self.settings
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(cfg['num_threads'])
        torch.use_deterministic_algorithms(True)
        self.device_ = torch.device(cfg['device'])
        self._fit_preprocessing(X_train)
        kwargs = ReferenceFTTransformer.get_default_kwargs(cfg['n_blocks'])
        kwargs.update(d_block=cfg['d_block'], attention_dropout=cfg['attention_dropout'],
                      ffn_dropout=cfg['ffn_dropout'])
        self.model_ = ReferenceFTTransformer(n_cont_features=len(self.num_cols),
            cat_cardinalities=[len(self.vocabs_[c]) + 1 for c in self.cat_cols],
            d_out=1, **kwargs).to(self.device_)
        if self.cat_cols:
            for embedding in self.model_.cat_embeddings.embeddings:
                embedding.padding_idx = 0
                with torch.no_grad():
                    embedding.weight[0].zero_()
        optimizer = torch.optim.AdamW(self.model_.make_parameter_groups(),
                                     lr=cfg['lr'], weight_decay=cfg['weight_decay'])
        loss_fn = torch.nn.BCEWithLogitsLoss()
        xn, xc = self._tensors(X_train)
        target = torch.as_tensor(y, dtype=torch.float32, device=self.device_)
        vn, vc = self._tensors(X_val)
        generator = torch.Generator().manual_seed(self.seed)
        self.history_ = []
        self.best_val_ap_, self.best_epoch_ = -np.inf, 0
        stale, best_state = 0, None
        for epoch in range(1, cfg['max_epochs'] + 1):
            self.model_.train()
            order = torch.randperm(len(y), generator=generator).to(self.device_)
            total_loss = 0.
            for ix in order.split(cfg['batch_size']):
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(None if xn is None else xn[ix],
                                     None if xc is None else xc[ix]).squeeze(-1)
                loss = loss_fn(logits, target[ix])
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(ix)
            ap = float(average_precision_score(yv, self._predict_tensors(vn, vc, len(yv))))
            self.history_.append(dict(epoch=epoch, train_loss=total_loss / len(y), val_ap=ap))
            if not np.isfinite(ap):
                raise RuntimeError('Validation AP is not finite')
            if ap > self.best_val_ap_:
                self.best_val_ap_, self.best_epoch_ = ap, epoch
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
            if cfg['verbose']:
                print(f'FTT epoch {epoch}: val AP={ap:.6f}, best={self.best_val_ap_:.6f}', flush=True)
            if stale >= cfg['patience']:
                break
        self.epochs_trained_ = len(self.history_)
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def _predict_tensors(self, xn, xc, n):
        import torch
        self.model_.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, n, self.settings['batch_size']):
                sl = slice(start, start + self.settings['batch_size'])
                logits = self.model_(None if xn is None else xn[sl],
                                     None if xc is None else xc[sl]).squeeze(-1)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.empty(0, dtype=np.float32)

    def predict_proba(self, X):
        if not hasattr(self, 'model_'):
            raise RuntimeError('Call fit before predict_proba')
        if not len(X):
            return np.empty(0, dtype=np.float32)
        return self._predict_tensors(*self._tensors(X), len(X))
