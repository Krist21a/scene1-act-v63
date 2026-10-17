#!/usr/bin/env python3
"""Verify the exported v63 files without importing robot-control code."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parent


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    failures = []
    frozen = json.loads((ROOT / 'diagnostics/full-visual-v63-hashes-20261017.json').read_text())
    payload = json.loads((ROOT / 'evidence/repository-sha256.json').read_text())
    for label, manifest in [('v63', frozen), ('repository', payload)]:
        for rel, expected in manifest.items():
            path = ROOT / rel
            if not path.is_file():
                failures.append(f'{label}: missing {rel}')
                continue
            with path.open('rb') as stream:
                if digest(stream) != expected:
                    failures.append(f'{label}: SHA mismatch {rel}')
    runs = json.loads((ROOT / 'evidence/original-results-manifest.json').read_text())
    for run, record in runs.items():
        with gzip.open(ROOT / 'results' / run / 'steps.jsonl.gz', 'rb') as stream:
            if digest(stream) != record['files']['steps.jsonl']['sha256']:
                failures.append(f'Log roundtrip: {run}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'OK: {len(frozen)} frozen v63 files, {len(payload)} repository files, {len(runs)} complete step logs.')


if __name__ == '__main__':
    main()
