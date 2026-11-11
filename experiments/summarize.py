"""Aggregate results/metrics.csv over seeds and plot metric vs. mask rate. OWNER: Vaibhav.

metrics.csv is append-only, so a (split, model, seed, mask_rate) run may appear more than
once; the LAST row wins. Writes:
  results/summary/summary.csv   mean, std, n_seeds per split / model / mask_rate / metric
  results/summary/summary.md    "mean ± std" tables, one per split (for the report)
  results/figures/<metric>_vs_mask_rate.png   one panel per split, one line per model,
                                              error bars = std over seeds

Usage:
  python -m experiments.summarize
  python -m experiments.summarize --metrics results/metrics.csv --plot ap auroc
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["auroc", "ap", "f1", "brier"]
LABELS = {"auroc": "AUROC", "ap": "Average precision", "f1": "F1", "brier": "Brier score"}
MODEL_ORDER = ["logreg", "xgb", "mlp", "ftt"]
MODEL_NAMES = {"logreg": "LogReg", "xgb": "XGBoost", "mlp": "MLP", "ftt": "FT-Transformer"}
SPLIT_ORDER = ["random", "chronological"]
KEYS = ["split", "model", "mask_rate"]


def _order(values, preferred):
    return [v for v in preferred if v in values] + sorted(set(values) - set(preferred))


def load_runs(path):
    df = pd.read_csv(path)
    missing = {"seed", *KEYS, *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    n = len(df)
    df = df.drop_duplicates(["seed", *KEYS], keep="last")
    if len(df) < n:
        print(f"dropped {n - len(df)} duplicate runs (kept the latest)")
    return df


def aggregate(df):
    g = df.groupby(KEYS)[METRICS]
    out = pd.concat({"mean": g.mean(), "std": g.std(ddof=1).fillna(0.0)}, axis=1)
    out.columns = [f"{m}_{s}" for s, m in out.columns]
    out["n_seeds"] = df.groupby(KEYS)["seed"].nunique()
    out = out.reset_index()
    out["split"] = pd.Categorical(out["split"], _order(out["split"].unique(), SPLIT_ORDER))
    out["model"] = pd.Categorical(out["model"], _order(out["model"].unique(), MODEL_ORDER))
    out = out.sort_values(KEYS, ignore_index=True)
    cols = KEYS + ["n_seeds"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
    return out[cols]


def to_markdown(summary):
    blocks = []
    for split, d in summary.groupby("split", observed=True):
        seeds = sorted(d["n_seeds"].unique())
        lines = [f"## {split} split (seeds per cell: {', '.join(map(str, seeds))})", "",
                 "| model | mask | " + " | ".join(LABELS[m] for m in METRICS) + " |",
                 "|---|---|" + "---|" * len(METRICS)]
        for r in d.itertuples(index=False):
            cells = [f"{getattr(r, m + '_mean'):.3f} ± {getattr(r, m + '_std'):.3f}" for m in METRICS]
            lines.append(f"| {MODEL_NAMES.get(r.model, r.model)} | {r.mask_rate:.0%} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    note = "Mean ± sample std over model seeds. Brier: lower is better; others: higher is better."
    return "# Results summary\n\n" + note + "\n\n" + "\n\n".join(blocks) + "\n"


def plot(summary, metric, out_dir):
    splits = list(summary["split"].cat.categories)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4), sharey=True, squeeze=False)
    for ax, split in zip(axes[0], splits):
        d = summary[summary["split"] == split]
        for model, dm in d.groupby("model", observed=True):
            ax.errorbar(dm["mask_rate"] * 100, dm[f"{metric}_mean"], yerr=dm[f"{metric}_std"],
                        marker="o", capsize=3, label=MODEL_NAMES.get(model, model))
        ax.set_title(f"{split} split")
        ax.set_xlabel("test cells masked (%)")
        ax.set_xticks(sorted(d["mask_rate"].unique() * 100))
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(LABELS[metric])
    axes[0][0].legend()
    n = int(summary["n_seeds"].max())
    bars = f"error bars: std over {n} seeds" if n > 1 else "single seed, no error bars"
    fig.suptitle(f"{LABELS[metric]} under test-time missingness ({bars})")
    fig.tight_layout()
    path = out_dir / f"{metric}_vs_mask_rate.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "results" / "metrics.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "summary"))
    ap.add_argument("--figures", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--plot", nargs="+", default=["ap"], choices=METRICS)
    args = ap.parse_args()

    out, figs = Path(args.out), Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    summary = aggregate(load_runs(args.metrics))
    summary.to_csv(out / "summary.csv", index=False)
    md = to_markdown(summary)
    (out / "summary.md").write_text(md)
    print(md)
    print(f"tables -> {out}")
    for metric in args.plot:
        print(f"figure -> {plot(summary, metric, figs)}")


if __name__ == "__main__":
    main()
