"""Tune baseline models using chronological validation AP only."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from src.data import load, split
from src.models import REGISTRY

ROOT = Path(__file__).resolve().parents[1]

SEARCH_SPACES = {
    "logreg": "C: log-uniform [0.001, 100]",
    "xgb": (
        "n_estimators: integer [100, 1000]; "
        "max_depth: integer [2, 8]; "
        "learning_rate: log-uniform [0.01, 0.3]; "
        "subsample: uniform [0.6, 1.0]; "
        "colsample_bytree: uniform [0.6, 1.0]; "
        "reg_lambda: log-uniform [0.1, 10]; "
        "min_child_weight: integer [1, 10]"
    ),
}


def log_uniform(rng, low, high):
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_config(name, rng):
    if name == "logreg":
        return {"C": log_uniform(rng, 0.001, 100)}

    if name == "xgb":
        return {
            "n_estimators": int(rng.integers(100, 1001)),
            "max_depth": int(rng.integers(2, 9)),
            "learning_rate": log_uniform(rng, 0.01, 0.3),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_lambda": log_uniform(rng, 0.1, 10),
            "min_child_weight": int(rng.integers(1, 11)),
        }

    raise ValueError(f"No search space for: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=list(SEARCH_SPACES)
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")

    X, y, num_cols, cat_cols = load()
    parts = split(X, y, "chronological", seed=args.seed)

    Xtr, ytr = parts["train"]
    Xva, yva = parts["val"]

    tuning_dir = ROOT / "results" / "tuning"
    config_dir = ROOT / "configs"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tuning_dir / f"{args.model}.csv"
    config_path = config_dir / f"{args.model}.yaml"

    rng = np.random.default_rng(args.seed)
    rows = []
    best_ap = -np.inf
    best_cfg = None
    best_trial = None

    for trial in range(1, args.trials + 1):
        cfg = sample_config(args.model, rng)
        model = REGISTRY[args.model](
            num_cols, cat_cols, seed=args.seed, **cfg
        )

        start = time.perf_counter()
        model.fit(Xtr, ytr, Xva, yva)
        fit_s = time.perf_counter() - start

        probabilities = model.predict_proba(Xva)
        val_ap = float(average_precision_score(yva, probabilities))

        if not np.isfinite(val_ap):
            raise ValueError(f"Invalid validation AP in trial {trial}")

        rows.append({
            "trial": trial,
            "model": args.model,
            "split": "chronological",
            "seed": args.seed,
            "val_ap": val_ap,
            "fit_s": fit_s,
            **cfg,
        })

        # Save progress after each completed trial.
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        if val_ap > best_ap:
            best_ap = val_ap
            best_cfg = cfg
            best_trial = trial

        print(
            f"Trial {trial}/{args.trials} | "
            f"validation AP: {val_ap:.6f} | "
            f"best AP: {best_ap:.6f}",
            flush=True,
        )

    comments = (
        f"# Search space: {SEARCH_SPACES[args.model]}\n"
        f"# Chronological split; seed {args.seed}; "
        f"{args.trials} trials; selected by validation AP.\n"
        f"# Best trial: {best_trial}; validation AP: {best_ap:.8f}\n"
    )
    config_path.write_text(
        comments + yaml.safe_dump(best_cfg, sort_keys=False)
    )

    print(f"\nBest config: {best_cfg}")
    print(f"Best validation AP: {best_ap:.6f}")
    print(f"Saved trials: {csv_path}")
    print(f"Saved config: {config_path}")


if __name__ == "__main__":
    main()