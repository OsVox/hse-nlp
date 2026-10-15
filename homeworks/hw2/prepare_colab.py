"""Prepare durable Colab inputs on CPU; restore them without network data fetches."""

from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import io

import pyarrow.parquet as pq

from .checkpoint_io import archive_checkpoint, drive_preflight, sha256
from .data import SHARDS, fetch_parquet, training_shard


def prepare(root: Path, cache: Path) -> None:
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('Prepare data on a CPU runtime before selecting a GPU')
    drive_preflight(root)
    prepared = root / 'prepared'
    prepared.mkdir(exist_ok=True)
    (prepared / 'ready.json').unlink(missing_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, path in enumerate(pool.map(lambda i: training_shard(i, cache), range(SHARDS)), 1):
            print(f'data_ready={index}/{SHARDS} {path.name}', flush=True)
    test = fetch_parquet('data/test/test.parquet', cache)
    questions = pq.read_table(test, columns=['question']).column('question').to_pylist()
    if len(questions) != 1919 or not all(isinstance(q, str) and q.strip() for q in questions):
        raise ValueError('Unexpected official test questions')
    question_hash = hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()
    files = {path.name: sha256(path) for path in sorted(cache.glob('*.parquet'))}
    for path in cache.glob('*.parquet'):
        pq.ParquetFile(path)
    local_archive = cache.parent / 'hw2_data.tar'
    with tarfile.open(local_archive, 'w') as bundle:
        for name in files:
            bundle.add(cache / name, arcname=name)
    _copy_verified(local_archive, prepared / 'data.tar')
    source_archive = cache.parent / 'hw2_source.tar'
    source = Path('homeworks/hw2')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    with tarfile.open(source_archive, 'w') as bundle:
        payload = (commit + '\n').encode()
        info = tarfile.TarInfo('source_commit.txt')
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
        for path in sorted(source.glob('*.py')):
            bundle.add(path, arcname=str(path))
        for path in sorted((source / 'tokenizer').glob('*')):
            if path.is_file():
                bundle.add(path, arcname=str(path))
    _copy_verified(source_archive, prepared / 'source.tar')
    archive_checkpoint(source / 'model', root / 'baseline' / 'historical-model')
    record = {
        'source_commit': commit,
        'data_sha256': sha256(prepared / 'data.tar'),
        'source_sha256': sha256(prepared / 'source.tar'),
        'parquet_files': files,
        'test_rows': len(questions),
        'questions_sha256': question_hash,
        'question_order': 'official data/test/test.parquet; original row order',
    }
    (prepared / 'ready.json').write_text(json.dumps(record, indent=2))
    print(f'CPU_PREPARATION_COMPLETE {prepared}', flush=True)


def _copy_verified(source: Path, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + '.partial')
    shutil.copyfile(source, partial)
    if sha256(source) != sha256(partial):
        raise IOError(f'Drive copy verification failed: {destination.name}')
    partial.replace(destination)


def restore(root: Path, repository: Path, cache: Path) -> None:
    drive_preflight(root)
    prepared = root / 'prepared'
    record = json.loads((prepared / 'ready.json').read_text())
    for name, destination in [('source', repository), ('data', cache)]:
        destination.mkdir(parents=True, exist_ok=True)
        local = Path('/content') / f'hw2_{name}_restore.tar'
        shutil.copyfile(prepared / f'{name}.tar', local)
        if sha256(local) != record[f'{name}_sha256']:
            raise IOError(f'Drive {name} bundle is corrupt')
        with tarfile.open(local) as bundle:
            bundle.extractall(destination, filter='data')
        local.unlink()
    for name, digest in record['parquet_files'].items():
        if sha256(cache / name) != digest:
            raise IOError(f'Restored data is corrupt: {name}')
    print(f'RESTORED_VERIFIED test_rows={record["test_rows"]}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=Path('/content/hw2_cache'))
    args = parser.parse_args()
    prepare(args.drive_root, args.cache_dir)
