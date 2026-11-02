"""Load, split, and preprocess UCI Bank Marketing (bank-additional-full.csv)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bank-additional-full.csv"
LEAKY = ["duration"]  # only known after the call ends


def load(path=DATA_PATH):
    """Return (X DataFrame, y int array, numeric cols, categorical cols). Row order = date order."""
    df = pd.read_csv(path, sep=";")
    y = (df.pop("y") == "yes").astype(int).to_numpy()
    X = df.drop(columns=LEAKY)
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return X, y, num_cols, cat_cols


def split(X, y, mode, seed=42):
    """70/15/15 split. mode='chronological' keeps source order; mode='random' stratifies on y.

    Returns dict with keys train/val/test -> (X, y).
    """
    if mode == "chronological":
        n = len(X)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        idx = np.arange(n)
        parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va], "test": idx[n_tr + n_va:]}
    elif mode == "random":
        idx = np.arange(len(X))
        tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
        parts = {"train": tr, "val": va, "test": te}
    else:
        raise ValueError(f"unknown split mode: {mode}")
    return {k: (X.iloc[v].reset_index(drop=True), y[v]) for k, v in parts.items()}


def make_preprocessor(num_cols, cat_cols):
    """Median-impute + standardize numerics; constant-impute + one-hot categoricals.

    Fit on TRAIN only. The dataset's own 'unknown' strings stay as a real category;
    artificially masked cells (NaN) map to '__missing__'.
    """
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


def summarize(splits):
    """Rows and positive rate per partition."""
    return pd.DataFrame(
        {k: {"rows": len(yv), "pos_rate": float(yv.mean())} for k, (_, yv) in splits.items()}
    ).T
