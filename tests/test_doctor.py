from contextlib import redirect_stdout
from hashlib import sha256
from importlib import import_module
from io import BytesIO, StringIO, TextIOWrapper
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gpt_animeimage_denoise import cli


PROJECT = Path(__file__).resolve().parents[1]


class DoctorTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.models = self.root / '空 模型'
        self.catalog = self.root / 'manifest.json'
        self.catalog.write_text(json.dumps({
            'v2/example.pt': {'sha256': sha256(b'exact').hexdigest()},
            'optional.pth': {'sha256': sha256(b'network').hexdigest(), 'optional': True},
        }), encoding='utf-8')

    def doctor(self, *options):
        from gpt_animeimage_denoise import doctor

        stdout = StringIO()
        with patch.object(doctor, 'MODEL_MANIFEST', self.catalog), redirect_stdout(stdout):
            result = cli.main(['doctor', '--json', '--model-dir', str(self.models), *options])
        return result, json.loads(stdout.getvalue())

    def test_whole_mode_probes_real_cpu_and_needs_no_weights(self):
        result, report = self.doctor('--whole', '--device', 'cpu')
        self.assertEqual(result, 0, report)
        self.assertEqual(report['device']['selected'], 'cpu')
        self.assertTrue(report['device']['tested'])
        self.assertEqual(report['python']['executable'], sys.executable)
        self.assertEqual(report['models']['missing'], [])
        self.assertFalse(self.models.exists())

    def test_missing_and_corrupt_model_files_are_diagnosed_without_writing(self):
        result, report = self.doctor('--device', 'cpu')
        self.assertEqual(result, 1)
        self.assertEqual(report['models']['missing'], ['v2/example.pt'])
        self.assertTrue(any('tools.assets fetch' in issue['repair'] for issue in report['issues']))
        weights = self.models / 'v2/example.pt'
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b'invalid')
        result, report = self.doctor('--device', 'cpu', '--verify-models')
        self.assertEqual(result, 1)
        self.assertEqual(report['models']['invalid'], ['v2/example.pt'])
        self.assertEqual(weights.read_bytes(), b'invalid')
        weights.write_bytes(b'exact')
        self.assertEqual(self.doctor('--device', 'cpu', '--verify-models')[0], 0)

    def test_valid_empty_model_files_are_not_reported_as_corrupt(self):
        catalog = json.loads(self.catalog.read_text(encoding='utf-8'))
        catalog['v2/__init__.py'] = {'sha256': sha256(b'').hexdigest()}
        self.catalog.write_text(json.dumps(catalog), encoding='utf-8')
        (self.models / 'v2').mkdir(parents=True)
        (self.models / 'v2/example.pt').write_bytes(b'exact')
        (self.models / 'v2/__init__.py').touch()
        for options in ([], ['--verify-models']):
            with self.subTest(options=options):
                result, report = self.doctor('--device', 'cpu', *options)
                self.assertEqual(result, 0, report)
                self.assertEqual(report['models']['checked'], 2)

    def test_broken_image_dependency_is_detected_even_when_metadata_exists(self):
        def import_runtime(name):
            if name == 'PIL.Image':
                raise ImportError('image extension failed to load')
            return import_module(name)

        with patch('gpt_animeimage_denoise.doctor.import_module', side_effect=import_runtime):
            result, report = self.doctor('--whole', '--device', 'cpu')
        self.assertEqual(result, 1)
        self.assertTrue(any(issue['code'] == 'image_runtime' and 'Pillow' in issue['message']
                            for issue in report['issues']))

    def test_requested_unavailable_cuda_reports_a_fix_instead_of_claiming_cpu(self):
        import torch

        with patch.object(torch.cuda, 'is_available', return_value=False):
            result, report = self.doctor('--whole', '--device', 'cuda')
        self.assertEqual(result, 1)
        self.assertIsNone(report['device']['selected'])
        self.assertTrue(any(issue['code'] == 'device' and '--device cpu' in issue['repair']
                            for issue in report['issues']))

    def test_stale_skill_record_reports_the_original_paths_and_reinstall_command(self):
        record = self.root / 'gaid-installation.json'
        old = self.root / 'moved'
        record.write_text(json.dumps({'project_root': str(old),
            'python_executable': str(old / 'python.exe'), 'model_dir': str(self.models)}), encoding='utf-8')
        code, report = self.doctor('--whole', '--device', 'cpu', '--installation', str(record))
        self.assertEqual(code, 1)
        self.assertEqual(report['installation']['project_root'], str(old))
        self.assertTrue(any(issue['code'] == 'skill_installation' and 'install_agent_skill.py' in issue['repair']
                            for issue in report['issues']))

    def test_stale_editable_install_is_reported_with_its_original_checkout(self):
        from gpt_animeimage_denoise import doctor

        original = self.root / '已经移动'
        distribution = SimpleNamespace(requires=[], read_text=lambda _: json.dumps({
            'url': original.as_uri(), 'dir_info': {'editable': True},
        }))
        with patch.object(doctor.metadata, 'distribution', return_value=distribution):
            result, report = self.doctor('--whole', '--device', 'cpu')
        self.assertEqual(result, 1)
        self.assertEqual(report['package']['editable_root'], str(original))
        self.assertTrue(any(issue['code'] == 'editable_install' for issue in report['issues']))

    def test_human_report_handles_a_redirected_windows_legacy_encoding(self):
        output = TextIOWrapper(BytesIO(), encoding='gbk')
        model_dir = self.root / 'models-\N{TEST TUBE}'
        with output, redirect_stdout(output):
            code = cli.main(['doctor', '--whole', '--device', 'cpu', '--model-dir', str(model_dir)])
            output.flush()
            self.assertEqual(code, 0)
            self.assertIn('诊断通过'.encode('gbk'), output.buffer.getvalue())

    def test_cli_can_report_a_broken_install_without_site_packages(self):
        probe = (
            'import runpy, sys\n'
            'sys.path.insert(0, sys.argv.pop(1))\n'
            'runpy.run_module("gpt_animeimage_denoise.cli", run_name="__main__")\n'
        )
        result = subprocess.run(
            [sys.executable, '-I', '-S', '-X', 'utf8', '-c', probe, str(PROJECT / 'src'),
             'doctor', '--whole', '--device', 'cpu', '--json'],
            cwd=self.root, capture_output=True, encoding='utf-8', timeout=15,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(any(issue['code'] == 'dependencies' for issue in report['issues']))
        self.assertTrue(any(issue['code'] == 'torch' for issue in report['issues']))


if __name__ == '__main__':
    unittest.main()
