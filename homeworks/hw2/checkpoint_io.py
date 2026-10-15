"""Verified checkpoint copies and exact JSONL packaging, without GPU work."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def drive_preflight(root: Path, *, mount: Path = Path('/content/drive')) -> None:
    """Fail before model allocation unless the intended Drive is mounted."""
    if not mount.is_mount() or not root.resolve().is_relative_to(mount.resolve()):
        raise RuntimeError('Mount Google Drive first and choose a path under /content/drive')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=root, prefix='.write-probe-', delete=False) as output:
        path = Path(output.name)
        output.write(b'HW2 Drive write/read verification\n')
    try:
        if path.read_bytes() != b'HW2 Drive write/read verification\n':
            raise IOError('Drive write/read verification failed')
    finally:
        path.unlink(missing_ok=True)


def archive_checkpoint(source: Path, destination: Path) -> dict:
    """Write an immutable copy, verify every file, then publish its marker."""
    marker = destination / 'checkpoint.complete.json'
    files = {p.name: sha256(p) for p in source.iterdir() if p.is_file()}
    if not files or 'model.safetensors' not in files:
        raise ValueError('Checkpoint has no model weights')
    if marker.exists():
        record = json.loads(marker.read_text())
        if record['files'] != files:
            raise FileExistsError('Refusing to overwrite a different checkpoint')
        if any(sha256(destination / name) != digest for name, digest in files.items()):
            raise IOError('Archived checkpoint failed integrity verification')
        return record
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in files.items():
        temporary = destination / (name + '.partial')
        shutil.copyfile(source / name, temporary)
        if sha256(temporary) != digest:
            raise IOError(f'Checkpoint copy failed verification: {name}')
        temporary.replace(destination / name)
    record = {'files': files, 'status': 'checkpoint_verified'}
    marker.write_text(json.dumps(record, indent=2))
    return record


def package_jsonl(csv_path: Path, output: Path, *, expected_rows: int = 1919) -> str:
    with csv_path.open(newline='', encoding='utf-8') as source:
        programs = [row['code'] for row in csv.DictReader(source)]
    if len(programs) != expected_rows or not all(p.strip() for p in programs):
        raise ValueError('Wrong row count or empty program')
    temporary = output.with_suffix(output.suffix + '.partial')
    with temporary.open('w', encoding='utf-8') as target:
        for code in programs:
            target.write(json.dumps({'code': code}, ensure_ascii=False) + '\n')
    restored = [json.loads(line)['code'] for line in temporary.read_text().splitlines()]
    if restored != programs:
        raise IOError('JSONL packaging changed a model output')
    temporary.replace(output)
    return sha256(output)
