"""Run models x splits x seeds x mask rates and append rows to results/metrics.csv.

Examples
    python -m experiments.run --models logreg xgb --seeds 42            # reproduce pilot
    python -m experiments.run --models logreg xgb mlp ftt --seeds 0 1 2 3 4
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from src.data import load, split, summarize
from src.masking import mask_inputs
from src.metrics import best_f1_threshold, evaluate
from src.models import REGISTRY

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["logreg", "xgb"], choices=list(REGISTRY))
    ap.add_argument("--splits", nargs="+", default=["random", "chronological"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--mask-rates", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--split-seed", type=int, default=42,
                    help="random split stays fixed across model seeds (chronological is always fixed)")
    ap.add_argument("--out", default=str(RESULTS / "metrics.csv"))
    args = ap.parse_args()

    X, y, num_cols, cat_cols = load()
    rows = []
    for mode in args.splits:
        parts = split(X, y, mode, seed=args.split_seed)
        print(f"\n== {mode} split ==\n{summarize(parts)}")
        (Xtr, ytr), (Xva, yva), (Xte, yte) = parts["train"], parts["val"], parts["test"]

        for name in args.models:
            for seed in args.seeds:
                model = REGISTRY[name](num_cols, cat_cols, seed=seed)
                t0 = time.perf_counter()
                model.fit(Xtr, ytr, Xva, yva)
                fit_s = time.perf_counter() - t0
                thr = best_f1_threshold(yva, model.predict_proba(Xva))

                for rate in args.mask_rates:
                    Xm = mask_inputs(Xte, rate)
                    t0 = time.perf_counter()
                    p = model.predict_proba(Xm)
                    infer_ms_per_row = 1000 * (time.perf_counter() - t0) / len(Xm)
                    row = {"split": mode, "model": name, "seed": seed, "mask_rate": rate,
                           "threshold": thr, **evaluate(yte, p, thr),
                           "fit_s": fit_s, "infer_ms_per_row": infer_ms_per_row}
                    rows.append(row)
                    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"\nwrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
