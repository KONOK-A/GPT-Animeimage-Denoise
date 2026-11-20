from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
import unittest
from unittest import mock

from packaging.requirements import Requirement


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'tools/switch_pytorch.py'
PINS = {'torch': '2.11.0', 'torchvision': '0.26.0'}


class SwitchPyTorchTest(unittest.TestCase):
    def setUp(self):
        self.venv = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(mock.patch.object(sys, 'prefix', str(self.venv)))

    def run_switch(self, target, run_result):
        self.assertTrue(SCRIPT.is_file(), 'PyTorch switch command is missing')
        entry = runpy.run_path(str(SCRIPT))['main']
        stdout, stderr = StringIO(), StringIO()
        with (mock.patch('subprocess.run', side_effect=run_result) as run,
              redirect_stdout(stdout), redirect_stderr(stderr)):
            code = entry([target])
        return code, [call.args[0] for call in run.call_args_list], stdout.getvalue(), stderr.getvalue()

    def runtime(self, tag, cuda_version=None, cuda_available=False):
        stdout = json.dumps({'torch': f'{PINS["torch"]}+{tag}',
                             'torchvision': f'{PINS["torchvision"]}+{tag}',
                             'cuda': cuda_version, 'cuda_available': cuda_available})
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr='')

    def test_downloads_exact_pair_then_installs_offline_and_verifies_cuda(self):
        success = subprocess.CompletedProcess([], 0)
        code, calls, output, errors = self.run_switch(
            'cuda', [success, success, self.runtime('cu128', '12.8', cuda_available=True)])
        self.assertEqual((code, errors), (0, ''))
        download, install, verify = calls
        self.assertIn('download', download)
        self.assertIn('https://download.pytorch.org/whl/cu128', download)
        wheels = download[download.index('--dest') + 1]
        self.assertIn('install', install)
        self.assertIn('--no-index', install)
        self.assertIn('--force-reinstall', install)
        self.assertEqual(install[install.index('--find-links') + 1], wheels)
        for name in ('torch', 'torchvision'):
            package = f'{name}=={PINS[name]}+cu128'
            self.assertIn(package, download)
            self.assertIn(package, install)
            self.assertIn(f'{PINS[name]}+cu128', output)
        self.assertIn('-c', verify)
        self.assertIn('CUDA 12.8', output)

    def test_download_failure_never_installs(self):
        code, calls, _, _ = self.run_switch('cpu', subprocess.CalledProcessError(1, ['pip', 'download']))
        self.assertNotEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIn('download', calls[0])
        self.assertIn(f'torch=={PINS["torch"]}+cpu', calls[0])

    def test_rejects_a_runtime_that_did_not_switch(self):
        success = subprocess.CompletedProcess([], 0)
        code, calls, output, errors = self.run_switch('cuda', [success, success, self.runtime('cpu')])
        self.assertNotEqual(code, 0)
        self.assertEqual(len(calls), 3)
        self.assertIn(f'{PINS["torch"]}+cpu', output)
        self.assertTrue(errors)

    def test_project_allows_the_official_windows_rocm_wheels(self):
        dependencies = tomllib.loads((PROJECT / 'pyproject.toml').read_text(encoding='utf-8'))['project']['dependencies']
        requirements = {req.name: req for req in map(Requirement, dependencies)}
        self.assertIn('2.9.1+rocm7.2.1', requirements['torch'].specifier)
        self.assertIn('0.24.1+rocm7.2.1', requirements['torchvision'].specifier)
