import subprocess
import sys
import tempfile
from pathlib import Path
import unittest


_DESKTOP = Path(__file__).resolve().parents[1] / 'src/gpt_animeimage_denoise/workbench/desktop.py'
_RUN_DESKTOP = '''
import ctypes
import runpy
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

site = Path(sys.argv[2])
sys.path.insert(0, str(site))
workbench = ModuleType('gpt_animeimage_denoise.workbench')
workbench.__path__ = [str(site / 'gpt_animeimage_denoise/workbench')]
sys.modules[workbench.__name__] = workbench
ctypes.windll = SimpleNamespace(user32=SimpleNamespace(
    MessageBoxW=lambda _window, text, _title, _flags: print(text),
))
runpy.run_path(sys.argv[1], run_name='__main__')
'''


class WorkbenchDesktopTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.site = self.root / 'site'
        self.log = self.root / 'outputs/workbench/workbench.log'

    def run_desktop(self):
        return subprocess.run(
            [sys.executable, '-I', '-S', '-X', 'utf8', '-c', _RUN_DESKTOP, str(_DESKTOP), str(self.site)],
            cwd=self.root, capture_output=True, encoding='utf-8', timeout=15,
        )

    def install_metadata(self, name, version, requirements=()):
        distribution = self.site / f'{name.replace("-", "_")}-{version}.dist-info'
        distribution.mkdir(parents=True)
        fields = f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n'
        fields += ''.join(f'Requires-Dist: {requirement}\n' for requirement in requirements)
        (distribution / 'METADATA').write_text(fields, encoding='utf-8')

    def install_server(self, source):
        workbench = self.site / 'gpt_animeimage_denoise/workbench'
        workbench.mkdir(parents=True)
        (workbench / '__init__.py').write_text('from .server import launch_workbench\n', encoding='utf-8')
        (workbench / 'server.py').write_text(source, encoding='utf-8')

    def assert_install_failure(self, result, dependency):
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertRegex(result.stdout, rf'\b{dependency}\b')
        self.assertIn('README.md', result.stdout)
        self.assertIn(r'.\.venv\Scripts\python.exe -m pip install -e .', result.stdout)
        self.assertIn('doctor.py', result.stdout)
        self.assertIn(dependency, self.log.read_text(encoding='utf-8'))

    def test_uninstalled_project_reports_missing_package(self):
        self.assert_install_failure(self.run_desktop(), 'gpt-animeimage-denoise')

    def test_missing_lazy_dependencies_are_reported_before_loading_server(self):
        self.install_metadata('gpt-animeimage-denoise', '1.0.0', ['torch==2.11.0', 'torchvision==0.26.0'])
        self.install_server("raise AssertionError('server loaded too early')")

        result = self.run_desktop()

        self.assert_install_failure(result, 'torch')
        self.assertRegex(result.stdout, r'\btorchvision\b')
        self.assertNotIn('server loaded too early', self.log.read_text(encoding='utf-8'))

    def test_server_import_error_includes_install_instructions(self):
        self.install_metadata('gpt-animeimage-denoise', '1.0.0')
        self.install_server('import missing_image_backend')

        self.assert_install_failure(self.run_desktop(), 'missing_image_backend')

    def test_healthy_launch_stays_silent_without_importing_torch(self):
        self.install_metadata('gpt-animeimage-denoise', '1.0.0', ['torch<=2.11.0,>=2.9.1'])
        self.install_metadata('torch', '2.9.1+rocm7.2.1')
        (self.site / 'torch.py').write_text("raise AssertionError('torch imported at startup')\n", encoding='utf-8')
        self.install_server('''
def _open_browser(address):
    print(f'Opened {address}')
    return True

def launch_workbench(*, workspace, browser):
    assert workspace.is_dir()
    browser('http://127.0.0.1:12345/')
    return 0
''')

        result = self.run_desktop()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        self.assertEqual(result.stderr, '')
        self.assertIn('Opened http://127.0.0.1:12345/', self.log.read_text(encoding='utf-8'))
