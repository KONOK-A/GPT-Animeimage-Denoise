"""Offline checks for the model asset provisioning commands."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.assets import __main__ as assets


def entry(data, **metadata):
    return {'sha256': sha256(data).hexdigest(), **metadata}


class AssetsTest(unittest.TestCase):
    def setUp(self):
        self.work = TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.dest = self.root / 'dest'
        self.manifest = {
            'v2/model.pt': entry(b'model'),
            'v2/LICENSE': entry(b'license'),
            'optional.pth': entry(b'optional', optional=True),
        }
        (self.source / 'v2').mkdir()
        (self.source / 'v2/model.pt').write_bytes(b'model')
        (self.source / 'v2/LICENSE').write_bytes(b'license')

    def test_import_checks_integrity_and_only_copies_locked_assets(self):
        (self.source / 'v2/experimental.pt').write_bytes(b'experiment')
        assets.import_assets(self.source, self.dest, self.manifest)
        self.assertEqual(sorted(p.name for p in (self.dest / 'v2').iterdir()), ['LICENSE', 'model.pt'])
        self.assertEqual(assets.verify(self.dest, self.manifest), 2)
        self.assertEqual(assets.import_assets(self.source, self.dest, self.manifest), 0)

    def test_import_validates_source_and_destination_before_copying(self):
        (self.source / 'v2/LICENSE').write_bytes(b'invalid')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch.*LICENSE'):
            assets.import_assets(self.source, self.dest, self.manifest)
        self.assertFalse(self.dest.exists())
        (self.source / 'v2/LICENSE').write_bytes(b'license')
        (self.dest / 'v2').mkdir(parents=True)
        (self.dest / 'v2/LICENSE').write_bytes(b'existing')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch.*LICENSE'):
            assets.import_assets(self.source, self.dest, self.manifest)
        self.assertEqual((self.dest / 'v2/LICENSE').read_bytes(), b'existing')
        self.assertFalse((self.dest / 'v2/model.pt').exists())

    def test_download_hash_mismatch_leaves_no_final_or_partial_file(self):
        self.manifest = {'model.pt': entry(b'expected', url='https://example.invalid/weights')}
        with patch.object(assets, 'download', side_effect=lambda item, path: path.write_bytes(b'bad')):
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch.*model.pt'):
                assets.fetch_assets(self.dest, self.manifest)
        self.assertEqual(list(self.dest.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
