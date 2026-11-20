import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gpt_animeimage_denoise.pipeline import _digest, _sha
from gpt_animeimage_denoise.workbench.documents import Documents
from gpt_animeimage_denoise.workbench.storage import StorageBusy, write_atomic


class AtomicWriteTest(unittest.TestCase):
    def test_stages_and_replaces_in_the_resolved_directory(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            logical = root / 'profile' / 'GAID'
            physical = root / 'redirected'
            physical.mkdir()
            resolve_path = Path.resolve

            def resolve(path, *, strict=False):
                if path == logical:
                    self.assertTrue(logical.is_dir())
                    return physical
                return resolve_path(path, strict=strict)

            with patch.object(Path, 'resolve', new=resolve):
                settings = logical / 'settings.json'
                write_atomic(settings, b'first')
                self.assertEqual((physical / settings.name).read_bytes(), b'first')
                write_atomic(settings, b'updated')
                self.assertEqual((physical / settings.name).read_bytes(), b'updated')

            self.assertFalse(settings.exists())
            self.assertEqual(list(physical.iterdir()), [physical / settings.name])


class WorkbenchStorageTest(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        settings = patch.dict(os.environ, GAID_WORKBENCH_SETTINGS=str(self.root / 'preferences.json'))
        settings.start()
        self.addCleanup(settings.stop)
        self.docs = Documents(self.root / 'work')
        self.addCleanup(self.docs.close)
        self.destination = self.root / 'custom'
        self.destination.mkdir()
        self.digest = 'a' * 64
        self.root_case = self.docs.workspace / 'cases' / self.digest
        self.docs.cache.claim(self.root_case.parent, self.digest)
        self.assertEqual(self.docs.cache_info()['bytes'], 0)
        self.generated = {Path('analysis/result.png'): b'generated-result',
                          Path('.cache') / ('evidence-' + 'b' * 64 + '.npz'): b'generated-evidence'}
        for name, data in self.generated.items():
            (self.root_case / name).parent.mkdir(exist_ok=True)
            (self.root_case / name).write_bytes(data)

    def test_cache_moves_and_clears_only_owned_files_and_resumes_after_interrupted_copy(self):
        private = [
            self.docs.root / self.digest / 'original.png',
            self.docs.root / self.digest / 'render-6' / 'result.png',
            self.docs.root / self.digest / 'draft.json',
            self.docs.root / self.digest / 'correction-record' / 'allow.png',
            self.docs.workspace / 'inputs' / self.digest / 'source.png',
            self.root_case / 'analysis' / 'notes.txt',
            self.root_case / '.cache' / 'personal.npz',
            self.root_case.parent / ('c' * 64) / 'analysis' / 'result.png',
            self.destination / 'personal.png']
        for path in private:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'user-content')
        self.assertEqual(self.docs.cache_info()['bytes'], sum(map(len, self.generated.values())))
        copy = shutil.copyfileobj
        def interrupted(source, dest, length):
            if Path(source.name).suffix == '.npz':
                raise OSError('interrupted copy')
            return copy(source, dest, length)
        with patch('gpt_animeimage_denoise.workbench.storage.shutil.copyfileobj', side_effect=interrupted):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                self.docs.change_preferences({'cache_dir': str(self.destination)})
        self.assertEqual(self.docs.preferences.read()['cache_dir'], '')
        self.assertTrue(all((self.root_case / name).read_bytes() == data for name, data in self.generated.items()))
        self.generated[Path('analysis/result.png')] = b'newer-render'
        (self.root_case / 'analysis/result.png').write_bytes(b'newer-render')
        self.docs.change_preferences({'cache_dir': str(self.destination)})
        info = self.docs.cache_info()
        target = Path(info['path']) / self.digest
        self.assertTrue(target.is_relative_to(self.destination))
        for name, data in self.generated.items():
            self.assertFalse((self.root_case / name).exists())
            self.assertEqual((target / name).read_bytes(), data)
        self.assertEqual(self.docs.clear_cache(), {'path': str(target.parent), 'bytes': 0,
                                                  'reclaimable_bytes': 0, 'released_bytes': info['bytes']})
        self.assertTrue(all(path.read_bytes() == b'user-content' for path in private))
        self.docs.change_preferences({'cache_dir': ''})
        self.assertEqual(self.docs.cache_info()['path'], str(self.root_case.parent))

    def test_cache_rejects_busy_or_managed_destinations_and_uses_per_workspace_directory(self):
        with self.docs.cache.processing():
            with self.assertRaises(StorageBusy):
                self.docs.change_preferences({'cache_dir': str(self.destination)})
            with self.assertRaises(StorageBusy):
                self.docs.clear_cache()
        with self.docs._inference, self.assertRaises(StorageBusy):
            self.docs.clear_cache()
        for protected in ('inputs', 'documents', 'corrections'):
            folder = self.docs.workspace / protected
            folder.mkdir(exist_ok=True)
            with self.assertRaisesRegex(ValueError, '目录'):
                self.docs.change_preferences({'cache_dir': str(folder)})
        self.docs.change_preferences({'cache_dir': str(self.destination)})
        with Documents(self.root / 'other') as other:
            self.assertNotEqual(self.docs.cache_info()['path'], other.cache_info()['path'])
            self.assertTrue(Path(other.cache_info()['path']).is_relative_to(self.destination))
            other.clear_cache()
            self.assertGreater(self.docs.cache_info()['bytes'], 0)

    def test_old_unmarked_workbench_case_is_recognized_and_its_extra_files_are_preserved(self):
        draft = self.docs.root / self.digest / 'draft.json'
        draft.parent.mkdir(parents=True)
        draft.write_text(json.dumps({'id': self.digest}), encoding='utf-8')
        (self.root_case / '.gaid-workbench-case').unlink()
        report = self.root_case / 'analysis' / 'report.json'
        report.write_text(json.dumps({'source_sha256': self.digest, 'profile': 'fusion', 'folder': 'analysis',
                                      'edits': {}}), encoding='utf-8')
        unrelated = self.root_case / 'notes.txt'
        unrelated.write_text('keep', encoding='utf-8')
        self.assertGreater(self.docs.cache_info()['bytes'], 0)
        self.assertEqual(self.docs.clear_cache()['bytes'], 0)
        self.assertTrue(draft.is_file())
        self.assertEqual(unrelated.read_text(encoding='utf-8'), 'keep')

    def test_compact_keeps_the_verified_grid_and_candidate_for_reopening(self):
        analysis = self.root_case / 'analysis'
        hashes = {}
        for name in ('original.png', 'person_labels.png', 'objects.png', 'unknown.png'):
            path = analysis / name
            path.write_bytes(name.encode('utf-8'))
            hashes[name] = _sha(path)
        candidate = {'source_sha256': self.digest, 'lambda': .0005}
        selected = self.root_case / '.cache' / ('l0-' + _digest(candidate) + '.npz')
        selected.write_bytes(b'exact candidate')
        report = analysis / 'report.json'
        report.write_text(json.dumps({'source_sha256': self.digest, 'profile': 'fusion', 'folder': 'analysis',
                                      'status': 'complete', 'edits': {}, 'output_hashes': hashes,
                                      'candidate_cache_identity': candidate, 'candidate_cache_stage': 'l0'}), encoding='utf-8')
        private = self.root_case / '.cache' / 'personal.npz'
        private.write_bytes(b'keep')
        redundant = self.root_case / '.cache' / ('l0-' + 'c' * 64 + '.npz')
        redundant.write_bytes(b'outdated candidate')
        expected = sum(map(len, self.generated.values())) + redundant.stat().st_size
        self.assertEqual(self.docs.cache_info()['reclaimable_bytes'], expected)

        # An incomplete grid should be left alone so it can still use its
        # intermediate data when recovering the analysis.
        (analysis / 'objects.png').write_bytes(b'corrupt')
        self.assertEqual(self.docs.clear_cache(mode='compact')['released_bytes'], 0)
        (analysis / 'objects.png').write_bytes(b'objects.png')
        result = self.docs.clear_cache(mode='compact')
        self.assertEqual(result['released_bytes'], expected)
        self.assertEqual(result['reclaimable_bytes'], 0)
        self.assertTrue(all(_sha(analysis / name) == digest for name, digest in hashes.items()))
        self.assertTrue(report.is_file())
        self.assertEqual(selected.read_bytes(), b'exact candidate')
        self.assertEqual(private.read_bytes(), b'keep')
        self.assertFalse(redundant.exists())
