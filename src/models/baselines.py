"""Logistic Regression and XGBoost baselines (pilot configs from the intermediate report)."""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import make_preprocessor
from src.models.base import BaseModel


class _SklearnPipelineModel(BaseModel):
    def _estimator(self):
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.pipe = Pipeline([
            ("prep", make_preprocessor(self.num_cols, self.cat_cols)),
            ("clf", self._estimator()),
        ])
        self.pipe.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X)[:, 1]


class LogReg(_SklearnPipelineModel):
    def _estimator(self):
        cfg = {"C": 1.0, "solver": "lbfgs", "max_iter": 2000, **self.config}
        return LogisticRegression(random_state=self.seed, **cfg)


class XGB(_SklearnPipelineModel):
    def _estimator(self):
        cfg = {
            "n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
            "tree_method": "hist", "n_jobs": 4, **self.config,
        }
        return XGBClassifier(random_state=self.seed, eval_metric="logloss", **cfg)
