"""Download a completed checkpoint's small evidence files through local gdrive.

Weights and optimizer states stay in Drive. Run verify_export afterward before
submitting the downloaded predictions; a successful transfer is not validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def download(folder_id: str, destination: Path):
    listing = subprocess.check_output([
        'gdrive', 'files', 'list', '--parent', folder_id, '--max', '100',
        '--skip-header', '--full-name', '--field-separator', '|'], text=True)
    files = []
    for line in listing.splitlines():
        file_id, name, kind, *_ = line.split('|')
        if kind == 'regular' and Path(name).suffix in {'.json', '.jsonl', '.csv'}:
            if Path(name).name != name or name in {'.', '..'}:
                raise ValueError('Unsafe Drive filename')
            files.append((file_id, name))
    names = [name for _, name in files]
    if len(set(names)) != len(names):
        raise ValueError('Ambiguous duplicate Drive names')
    if not {'checkpoint.complete.json', 'results.complete.json',
            'predictions.jsonl', 'predictions.csv', 'predictions.provenance.json'} <= set(names):
        raise ValueError('Checkpoint results are not complete in Drive')
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    for file_id, name in files:
        subprocess.run(['gdrive', 'files', 'download', file_id,
                        '--destination', str(destination)], check=True)
        path = destination/name
        records.append(dict(id=file_id, name=name, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    manifest = dict(drive_folder_id=folder_id, files=records,
                    method='Authenticated local gdrive CLI; exported from Drive')
    (destination/'download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder_id')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    download(args.folder_id, args.destination)
