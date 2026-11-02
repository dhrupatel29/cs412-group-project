"""Pranay's isolated FT-Transformer evaluator and validation-only random search.

Does not modify Dhru's runner or results/metrics.csv. Tuning never reads test data.
python -m experiments.ftt tune --trials 20
python -m experiments.ftt evaluate --config configs/ft_transformer.yaml
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from src.data import load, split
from src.masking import mask_inputs
from src.metrics import best_f1_threshold, evaluate
from src.models.ft_transformer import FTTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['tune', 'evaluate'])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--config')
    parser.add_argument('--out-dir', default='results/pranay')
    parser.add_argument('--max-epochs', type=int)
    parser.add_argument('--patience', type=int)
    args = parser.parse_args()
    if args.trials < 1 or (args.max_epochs is not None and args.max_epochs < 1) or (args.patience is not None and args.patience < 1):
        parser.error('trials, max-epochs, and patience must be positive')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y, nums, cats = load()
    manifest = dict(cpu=platform.processor(), platform=platform.platform(),
                    logical_cpus=os.cpu_count(), torch=torch.__version__, device='cpu',
                    threads=4, model_seed=42, split_seed=42,
                    timing='fit includes preprocessing and validation early stopping; inference includes preprocessing; full-test batch; one warm-up prediction per condition')
    if args.mode == 'tune':
        # Slicing occurs before the chronological test boundary. No test partition is read.
        ntrain, nval = int(.70 * len(X)), int(.15 * len(X))
        Xtr, ytr = X.iloc[:ntrain], y[:ntrain]
        Xva, yva = X.iloc[ntrain:ntrain+nval], y[ntrain:ntrain+nval]
        rng = np.random.default_rng(42)
        rows = []
        best = -np.inf
        for trial in range(args.trials):
            cfg = dict(n_blocks=int(rng.integers(1,5)), d_block=int(rng.choice([96,128,192,256])),
                       attention_dropout=float(rng.uniform(.1,.4)), ffn_dropout=float(rng.uniform(0,.3)),
                       lr=float(10**rng.uniform(np.log10(3e-5),np.log10(3e-4))),
                       weight_decay=float(10**rng.uniform(-6,-4)), max_epochs=args.max_epochs or 100,
                       patience=args.patience or 16, device='cpu', num_threads=4)
            model = FTTransformer(nums, cats, seed=42, **cfg)
            start = time.perf_counter()
            model.fit(Xtr,ytr,Xva,yva)
            rows.append(dict(trial=trial+1, val_ap=model.best_val_ap_, best_epoch=model.best_epoch_,
                             epochs_trained=model.epochs_trained_, fit_s=time.perf_counter()-start,
                             **cfg))
            pd.DataFrame(rows).to_csv(out/'ftt_tuning.csv',index=False)
            if model.best_val_ap_ > best:
                best, best_cfg = model.best_val_ap_, cfg
                (out/'ftt_best_so_far.yaml').write_text(yaml.safe_dump(best_cfg,sort_keys=False))
            print(f'Trial {trial+1}/{args.trials}: AP={model.best_val_ap_:.6f}, best={best:.6f}, epochs={model.epochs_trained_}',flush=True)
        if args.trials == 20:
            target = Path('configs/ft_transformer.yaml')
            target.parent.mkdir(exist_ok=True)
            target.write_text('# 20 random-search trials, chronological validation AP, split/model seed 42.\n'
                '# Ranges: blocks 1-4; width 96/128/192/256; attention dropout .1-.4; FFN dropout 0-.3;\n'
                '# lr log-uniform 3e-5..3e-4; weight decay log-uniform 1e-6..1e-4. No test selection.\n'
                + yaml.safe_dump(best_cfg,sort_keys=False))
        manifest.update(completed_trials=args.trials, best_val_ap=best, test_used=False)
    else:
        cfg = yaml.safe_load(Path(args.config).read_text()) if args.config else {}
        if args.max_epochs is not None:
            cfg['max_epochs'] = args.max_epochs
        if args.patience is not None:
            cfg['patience'] = args.patience
        cfg['verbose'] = True
        manifest['effective_config'] = cfg
        manifest['configuration_source'] = args.config or 'class defaults'
        manifest['phase'] = 'development evaluation, not the final five-seed experiment'
        rows=[]
        for mode in ['random','chronological']:
            parts=split(X,y,mode,seed=42)
            Xtr,ytr=parts['train'];Xva,yva=parts['val'];Xte,yte=parts['test']
            model=FTTransformer(nums,cats,seed=42,**cfg)
            start=time.perf_counter();model.fit(Xtr,ytr,Xva,yva);fit_s=time.perf_counter()-start
            threshold=best_f1_threshold(yva,model.predict_proba(Xva))
            pd.DataFrame(model.history_).to_csv(out/f'ftt_{mode}_history.csv',index=False)
            for rate in [0.,.1,.2,.3]:
                Xm=mask_inputs(Xte,rate);model.predict_proba(Xm)
                start=time.perf_counter();p=model.predict_proba(Xm)
                elapsed=time.perf_counter()-start
                rows.append(dict(split=mode,model='ftt',seed=42,mask_rate=rate,
                    threshold=threshold,**evaluate(yte,p,threshold),fit_s=fit_s,
                    infer_ms_per_row=1000*elapsed/len(Xm)))
            pd.DataFrame(rows).to_csv(out/'ftt_metrics.csv',index=False)
            print(pd.DataFrame(rows).tail(4).to_string(index=False),flush=True)
    (out/f'ftt_{args.mode}_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
