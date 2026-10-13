import gc
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch
import revision_runner as runner
from io_utils import read_latest_rows, write_csv

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ['dataset', 'seed', 'alpha', 'beta', 'delta', 'status', 'error',
          'started_at', 'finished_at', 'log_file', 'acc1', 'nmi1', 'pur1',
          'ari1', 'f_score1']
_STRUCTURAL_NEIGHBORS = {'MSRC': 9, 'Notting-Hill': 9, 'Caltech7': 5, '100Leaves': 8,
             'COIL20': 4, 'uci-digit': 8, 'Digit-Product': 19, 'Hdigit': 19}


def run(name, entry, output, dry_run=False):
    data_file = ROOT / 'datasets' / entry['data_file']
    if not data_file.is_file():
        raise FileNotFoundError(data_file)
    print(f'{name}: {len(entry["seeds"])} runs; delta={entry["training"]["delta"]}', flush=True)
    if dry_run:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = dict(entry['training'], _structural_neighbors=_STRUCTURAL_NEIGHBORS[name])
    args = SimpleNamespace(dataset_dir=ROOT / 'datasets', **entry['runtime_options'])
    runner.ensure_manifest(output, dict(dataset=name, seeds=entry['seeds'],
                                      training=entry['training'], runtime_options=entry['runtime_options']))
    records = {int(row['seed']): row for row in read_latest_rows(output / 'runs.csv')}
    failures = 0
    for seed in entry['seeds']:
        if records.get(seed, {}).get('status') == 'completed':
            continue
        run_dir = output / f'seed{seed}'
        run_dir.mkdir(exist_ok=True)
        row = dict(dataset=name, seed=seed, alpha=cfg['alpha'], beta=cfg['beta'],
                   delta=cfg['delta'], status='running', error='',
                   started_at=runner.core.now_iso(), log_file=str(run_dir / 'train.log'))
        records[seed] = row
        write_csv(output / 'runs.csv', FIELDS, list(records.values()))
        prepared = None
        try:
            with (run_dir / 'train.log').open('a', encoding='utf-8', buffering=1) as stream, \
                 redirect_stdout(runner.core.Tee(sys.stdout, stream)), redirect_stderr(runner.core.Tee(sys.stderr, stream)):
                prepared = runner.prepare_training(name, seed, cfg, args, run_dir,
                                                   checkpoint=run_dir / 'pretrain.pt')
                row.update(runner.train(name, seed, cfg, args, prepared))
            row['status'] = 'completed'
            print(f'{name} seed={seed}: ACC={row["acc1"]:.4f}', flush=True)
        except KeyboardInterrupt:
            row['status'] = 'interrupted'
            raise
        except Exception as exc:
            failures += 1
            row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            with (run_dir / 'train.log').open('a', encoding='utf-8') as stream:
                traceback.print_exc(file=stream)
        finally:
            row['finished_at'] = runner.core.now_iso()
            write_csv(output / 'runs.csv', FIELDS, list(records.values()))
            del prepared
            gc.collect()
            torch.cuda.empty_cache()
    if failures:
        raise SystemExit(1)
