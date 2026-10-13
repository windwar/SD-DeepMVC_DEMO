import csv
import os
import time
from pathlib import Path

def write_csv(path, fields, rows, retries=3, retry_seconds=1.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.tmp')
    with pending.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(retries + 1):
        try:
            os.replace(pending, path)
            return True
        except PermissionError:
            if attempt < retries:
                time.sleep(retry_seconds)
                continue
            print(f'[SAVE PENDING] {path.name} is locked. Latest results are safe in '
                  f'{pending}; training continues. Close the CSV to allow the next save.', flush=True)
            return False

def read_latest_rows(path):
    path = Path(path)
    pending = path.with_suffix(path.suffix + '.tmp')
    def read(p):
        if not p.exists():
            return []
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f'Invalid CSV snapshot: {p}')
        return rows
    published = read(path)
    if pending.exists() and (not path.exists() or pending.stat().st_mtime_ns >= path.stat().st_mtime_ns):
        candidate = read(pending)
        def key(row):
            return tuple(row.get(k, '') for k in ['dataset', 'seed', 'alpha', 'beta'])
        old = {key(row): row for row in published}
        new = {key(row): row for row in candidate}
        if not set(old).issubset(new):
            raise ValueError(f'Pending snapshot omits existing runs: {pending}')
        if any(row.get('status') == 'completed' and new[k].get('status') != 'completed'
               for k, row in old.items()):
            raise ValueError(f'Pending snapshot would discard completed runs: {pending}')
        print(f'[RECOVERY] Loaded {len(candidate)} rows from {pending.name}.', flush=True)
        return candidate
    return published
