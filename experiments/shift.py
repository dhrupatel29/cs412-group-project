"""Temporal shift analysis of the chronological split. OWNER: Vaibhav.

Needs only the data. Writes:
  results/shift/prevalence.csv     rows + positive rate per partition (both splits)
  results/shift/numeric_drift.csv  train vs test mean, standardized mean diff, KS per numeric column
  results/shift/categorical_drift.csv  total variation distance train vs test per categorical column
  results/shift/category_props.csv     train/val/test proportions for month, poutcome, contact
  results/shift/summary.md         the same tables in Markdown, for the report
  results/figures/shift_euribor_over_time.png
  results/figures/shift_categorical.png

Usage: python -m experiments.shift
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import load, split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "shift"
FIGS = ROOT / "results" / "figures"

FOCUS_NUM = ["euribor3m", "nr.employed", "emp.var.rate", "cons.conf.idx"]
FOCUS_CAT = ["month", "poutcome", "contact"]
MONTHS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
PARTS = ["train", "val", "test"]


def to_markdown(df):
    """Markdown table without the optional 'tabulate' dependency."""
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def prevalence(X, y):
    rows = []
    for mode in ["random", "chronological"]:
        for part, (_, yp) in split(X, y, mode).items():
            rows.append({"split": mode, "partition": part, "rows": len(yp),
                         "positives": int(yp.sum()), "pos_rate": float(yp.mean())})
    return pd.DataFrame(rows)


def numeric_drift(parts, num_cols):
    tr, va, te = (parts[p][0] for p in PARTS)
    rows = []
    for col in num_cols:
        a, b = tr[col].to_numpy(float), te[col].to_numpy(float)
        ks = ks_2samp(a, b)
        sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({
            "feature": col,
            "train_mean": a.mean(), "val_mean": va[col].mean(), "test_mean": b.mean(),
            "smd": (b.mean() - a.mean()) / sd if sd > 0 else np.nan,
            "ks_stat": ks.statistic, "ks_p": ks.pvalue,
            "focus": col in FOCUS_NUM,
        })
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False, ignore_index=True)


def categorical_drift(parts, cat_cols):
    tr, te = parts["train"][0], parts["test"][0]
    rows = []
    for col in cat_cols:
        p = tr[col].value_counts(normalize=True)
        q = te[col].value_counts(normalize=True)
        p, q = p.align(q, fill_value=0.0)
        top = (q - p).abs().idxmax()
        rows.append({"feature": col, "tvd": 0.5 * (p - q).abs().sum(),
                     "largest_change": top, "train_prop": p[top], "test_prop": q[top],
                     "focus": col in FOCUS_CAT})
    return pd.DataFrame(rows).sort_values("tvd", ascending=False, ignore_index=True)


def category_props(parts):
    frames = []
    for col in FOCUS_CAT:
        props = pd.DataFrame({p: parts[p][0][col].value_counts(normalize=True) for p in PARTS}).fillna(0.0)
        if col == "month":
            props = props.reindex([m for m in MONTHS if m in props.index])
        props.index.name = "value"
        frames.append(props.reset_index().assign(feature=col))
    return pd.concat(frames, ignore_index=True)[["feature", "value", *PARTS]]


def plot_euribor(X, y, bounds):
    idx = np.arange(len(X))
    rate = pd.Series(y).rolling(1000, center=True, min_periods=200).mean()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    ax1.plot(idx, X["euribor3m"], lw=0.8, color="tab:blue")
    ax1.set_ylabel("euribor3m (%)")
    ax2.plot(idx, rate, lw=1.2, color="tab:red")
    ax2.set_ylabel("positive rate\n(rolling 1000 rows)")
    ax2.set_xlabel("row index (chronological order, May 2008 to Nov 2010)")
    for ax in (ax1, ax2):
        for b in bounds:
            ax.axvline(b, color="k", ls="--", lw=1)
        ax.grid(alpha=0.3)
    top = ax1.get_ylim()[1]
    for (lo, hi), name in zip([(0, bounds[0]), (bounds[0], bounds[1]), (bounds[1], len(X))], PARTS):
        ax1.text((lo + hi) / 2, top, name, ha="center", va="bottom", fontsize=10)
    fig.suptitle("Chronological split: interest rate and target rate over time", y=1.0)
    fig.tight_layout()
    path = FIGS / "shift_euribor_over_time.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_categorical(props):
    fig, axes = plt.subplots(1, len(FOCUS_CAT), figsize=(13, 3.8),
                             gridspec_kw={"width_ratios": [3, 1.3, 1]})
    for ax, col in zip(axes, FOCUS_CAT):
        d = props[props.feature == col].set_index("value")[PARTS]
        d.plot.bar(ax=ax, width=0.8, legend=col == FOCUS_CAT[0])
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("proportion" if col == FOCUS_CAT[0] else "")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Category proportions by chronological partition")
    fig.tight_layout()
    path = FIGS / "shift_categorical.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    X, y, num_cols, cat_cols = load()
    parts = split(X, y, "chronological")
    n_tr, n_va = len(parts["train"][1]), len(parts["val"][1])

    tables = {
        "prevalence": prevalence(X, y),
        "numeric_drift": numeric_drift(parts, num_cols),
        "categorical_drift": categorical_drift(parts, cat_cols),
        "category_props": category_props(parts),
    }
    md = ["# Temporal shift (chronological split)\n",
          "pdays uses 999 for 'never contacted', so its mean is not meaningful; KS still is.\n"]
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        md += [f"\n## {name}\n", to_markdown(df), ""]
        print(f"\n== {name} ==\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}")
    (OUT / "summary.md").write_text("\n".join(md))

    figs = [plot_euribor(X, y, (n_tr, n_tr + n_va)), plot_categorical(tables["category_props"])]
    print(f"\ntables -> {OUT}")
    for f in figs:
        print(f"figure -> {f}")


if __name__ == "__main__":
    main()
