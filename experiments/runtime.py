"""Pranay: summarize fit/inference measurements without counting each mask as a fit.

Example: python -m experiments.runtime --input results/metrics.csv
This summarizes existing timings. Comparable hardware/protocol must be verified
by the person producing the measurements; aggregation cannot establish it.
"""
import argparse
import json
import platform
from pathlib import Path
import pandas as pd


def summarize_runtime(df):
    keys = ['split', 'model', 'seed', 'mask_rate']
    required = keys + ['fit_s', 'infer_ms_per_row']
    if not set(required) <= set(df.columns):
        raise ValueError(f'Required columns: {required}')
    if df[required].isna().any().any() or (df[['fit_s', 'infer_ms_per_row']] < 0).any().any():
        raise ValueError('Timings and grouping columns must be present and nonnegative')
    if df.duplicated(keys).any():
        raise ValueError('Duplicate split/model/seed/mask rows: select one run before aggregation')
    # All masks reuse the same fit. Clean-only latency makes the summary comparable.
    clean = df[df.mask_rate == 0].copy()
    if clean.empty:
        raise ValueError('No clean (mask_rate=0) rows for the runtime table')
    out = clean.groupby(['split', 'model'], as_index=False).agg(
        n_seeds=('seed', 'nunique'), fit_s_mean=('fit_s', 'mean'),
        fit_s_std=('fit_s', 'std'), infer_ms_per_row_mean=('infer_ms_per_row', 'mean'),
        infer_ms_per_row_std=('infer_ms_per_row', 'std'))
    # A one-run standard deviation is unknown, not zero.
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', nargs='+', default=['results/metrics.csv'])
    ap.add_argument('--out', default='results/pranay/runtime_summary.csv')
    ap.add_argument('--measurement-manifest', help='Hardware/protocol JSON from the actual measuring process')
    ap.add_argument('--latest-per-condition', action='store_true', help='Explicitly select the last appended row per split/model/seed/mask from repeated pilot runs')
    args = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in args.input], ignore_index=True)
    if args.latest_per_condition:
        df = df.drop_duplicates(['split', 'model', 'seed', 'mask_rate'], keep='last')
    result = summarize_runtime(df)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest, index=False)
    lines = ['# Runtime summary (clean inputs only)', '',
             'Mask conditions share a fit and are not independent timing replicates.',
             'A blank standard deviation means only one seed was available.', '',
             '| Split | Model | Seeds | Fit seconds, mean (SD) | Inference ms/row, mean (SD) |',
             '|---|---|---:|---:|---:|']
    for r in result.itertuples():
        fmt = lambda v: 'N/A' if pd.isna(v) else f'{v:.6f}'
        lines.append(f'| {r.split} | {r.model} | {r.n_seeds} | {fmt(r.fit_s_mean)} ({fmt(r.fit_s_std)}) | {fmt(r.infer_ms_per_row_mean)} ({fmt(r.infer_ms_per_row_std)}) |')
    if args.measurement_manifest:
        m = json.loads(Path(args.measurement_manifest).read_text())
        lines += ['', 'Measurement manifest:', '```json', json.dumps(m, indent=2), '```']
    else:
        lines += ['', 'No measurement manifest supplied. Hardware and timing comparability are not verified.']
    lines += ['', 'This script does not claim to benchmark models that are absent from the inputs.']
    if args.latest_per_condition:
        lines += ['For duplicate pilot runs, the last appended row per condition was explicitly selected.']
    dest.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
