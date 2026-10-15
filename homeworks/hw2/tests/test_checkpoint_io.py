"""Exercise durable-copy failure handling and byte-preserving packaging."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from homeworks.hw2.checkpoint_io import archive_checkpoint, drive_preflight, package_jsonl


class CheckpointIOTests(unittest.TestCase):
    def test_copy_is_immutable_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source'; source.mkdir()
            (source/'model.safetensors').write_bytes(b'weights')
            destination = root/'archive'
            first = archive_checkpoint(source, destination)
            self.assertEqual(archive_checkpoint(source, destination), first)
            (destination/'model.safetensors').write_bytes(b'corrupt')
            with self.assertRaises(IOError): archive_checkpoint(source, destination)
            (source/'model.safetensors').write_bytes(b'different weights')
            with self.assertRaises(FileExistsError): archive_checkpoint(source, destination)

    def test_packaging_keeps_program_whitespace_and_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            programs=['def solve():\n    return 1\n', '  x = "é"\n']
            with (root/'out.csv').open('w', newline='', encoding='utf-8') as file:
                writer=csv.DictWriter(file,fieldnames=['code']); writer.writeheader()
                writer.writerows({'code': code} for code in programs)
            package_jsonl(root/'out.csv',root/'out.jsonl',expected_rows=2)
            self.assertEqual([json.loads(line)['code'] for line in (root/'out.jsonl').read_text().splitlines()], programs)

    def test_unmounted_drive_fails_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(RuntimeError): drive_preflight(root/'runs',mount=root)


if __name__ == '__main__': unittest.main()
