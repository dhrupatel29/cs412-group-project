

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from src.data import load, split, summarize
from src.masking import mask_inputs
from src.metrics import best_f1_threshold, evaluate
from src.models import REGISTRY

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIGS = ROOT / "configs"


def load_config(name):
    """Read model settings, or use defaults if no config exists."""
    cfg_path = CONFIGS / f"{name}.yaml"

    if not cfg_path.exists():
        print(f"{name}: using default settings")
        return {}

    cfg = yaml.safe_load(cfg_path.read_text())

    if cfg is None:
        cfg = {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must contain a mapping: {cfg_path}")

    print(f"{name}: loaded settings from {cfg_path.name}")
    return cfg


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--models",
        nargs="+",
        default=["logreg", "xgb"],
        choices=list(REGISTRY),
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["random", "chronological"],
        choices=["random", "chronological"],
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )
    ap.add_argument(
        "--mask-rates",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3],
    )
    ap.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Keep the random split fixed across model seeds.",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS / "metrics.csv"),
    )

    args = ap.parse_args()

    for rate in args.mask_rates:
        if not 0.0 <= rate <= 1.0:
            ap.error("--mask-rates must be between 0 and 1")

    X, y, num_cols, cat_cols = load()

    pred_dir = RESULTS / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        name: load_config(name)
        for name in args.models
    }

    rows = []

    for mode in args.splits:
        parts = split(X, y, mode, seed=args.split_seed)

        print(f"\n== {mode} split ==")
        print(summarize(parts))

        Xtr, ytr = parts["train"]
        Xva, yva = parts["val"]
        Xte, yte = parts["test"]

        for name in args.models:
            for seed in args.seeds:
                model = REGISTRY[name](
                    num_cols,
                    cat_cols,
                    seed=seed,
                    **configs[name],
                )

                t0 = time.perf_counter()
                model.fit(Xtr, ytr, Xva, yva)
                fit_s = time.perf_counter() - t0

                val_predictions = model.predict_proba(Xva)
                threshold = best_f1_threshold(yva, val_predictions)

                for rate in args.mask_rates:
                    Xm = mask_inputs(Xte, rate)

                    t0 = time.perf_counter()
                    p = model.predict_proba(Xm)
                    infer_ms_per_row = (
                        1000 * (time.perf_counter() - t0) / len(Xm)
                    )

                    pred_path = (
                        pred_dir
                        / f"{mode}_{name}_seed{seed}_mask{rate}.csv"
                    )

                    pd.DataFrame({
                        "y": yte,
                        "p": p,
                    }).to_csv(pred_path, index=False)

                    row = {
                        "split": mode,
                        "model": name,
                        "seed": seed,
                        "mask_rate": rate,
                        "threshold": threshold,
                        **evaluate(yte, p, threshold),
                        "fit_s": fit_s,
                        "infer_ms_per_row": infer_ms_per_row,
                    }

                    rows.append(row)

                    print({
                        key: round(value, 4)
                        if isinstance(value, float)
                        else value
                        for key, value in row.items()
                    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    needs_header = not out.exists() or out.stat().st_size == 0

    df.to_csv(
        out,
        mode="a",
        header=needs_header,
        index=False,
    )

    print(f"\nwrote {len(df)} rows -> {out}")
    print(f"predictions saved -> {pred_dir}")


if __name__ == "__main__":
    main()