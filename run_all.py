import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def validate(plan):
    entries = plan['datasets']
    if len(entries) != 8 or sum(len(e['seeds']) for e in entries.values()) != 160:
        raise ValueError('Expected exactly 8 datasets and 160 runs')
    for name, entry in entries.items():
        if len(entry['seeds']) != 20 or len(set(entry['seeds'])) != 20:
            raise ValueError('Expected 20 unique seeds: ' + name)

def worker(args, plan):
    sys.path.insert(0, str(ROOT / 'code'))
    import experiment
    entry = plan['datasets'][args.worker]
    experiment.run(args.worker, entry, args.output / args.worker, args.dry_run)

def summarize(plan, output):
    sys.path.insert(0, str(ROOT / 'code'))
    from io_utils import read_latest_rows, write_csv
    runs, summaries = [], []
    for name, entry in plan['datasets'].items():
        rows = read_latest_rows(output / name / 'runs.csv')
        expected = set(entry['seeds'])
        rows = [r for r in rows if r['dataset'] == name and int(r['seed']) in expected]
        if len({int(r['seed']) for r in rows}) != len(rows):
            raise ValueError('Duplicate result seeds: ' + name)
        public_rows = []
        for row in rows:
            row = dict(row)
            public_rows.append(row)
        runs.extend(public_rows)
        done = [r for r in rows if r['status'] == 'completed']
        summary = dict(dataset=name, completed=len(done), expected=20,
                       status='completed' if len(done) == 20 else 'pending',
                       delta=entry['training']['delta'],
                       seeds=';'.join(map(str, entry['seeds'])))
        for metric in ['acc1', 'nmi1', 'pur1', 'ari1', 'f_score1']:
            values = [float(r[metric]) for r in done]
            summary[metric + '_mean'] = statistics.mean(values) if values else ''
            summary[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    if runs:
        write_csv(output / 'runs.csv', list(runs[0]), runs)
    write_csv(output / 'summary.csv', list(summaries[0]), summaries)
    return sum(r['completed'] for r in summaries)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Validate all 160 planned runs without training')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    plan = json.loads((ROOT / 'experiment_plan.json').read_text(encoding='utf-8'))
    if args.worker:
        worker(args, plan)
        return
    validate(plan)
    print('Verified package: 8 datasets x 20 configured seeds = 160 runs', flush=True)
    failures = []
    for name in plan['datasets']:
        command = [sys.executable, '-I', str(ROOT / 'run_all.py'),
                   '--worker', name, '--output', str(args.output)]
        if args.dry_run:
            command.append('--dry-run')
        result = subprocess.run(command, cwd=str(ROOT))
        if result.returncode:
            failures.append(name)
        if not args.dry_run:
            print('Overall completed: {}/160'.format(summarize(plan, args.output)), flush=True)
    if failures:
        raise SystemExit('Failed datasets: ' + ', '.join(failures) + '; rerun the same command to retry')
    if not args.dry_run and summarize(plan, args.output) != 160:
        raise SystemExit('Not all runs completed; inspect dataset logs and rerun')

if __name__ == '__main__':
    main()
