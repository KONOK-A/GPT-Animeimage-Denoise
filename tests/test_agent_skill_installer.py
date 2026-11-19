from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT / 'tools/install_agent_skill.py'
SKILL = 'gpt-animeimage-denoise'
SOURCE = PROJECT / '.agents/skills' / SKILL / 'SKILL.md'


class AgentSkillInstallerTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.home = self.root / 'user'
        self.claude_dir = self.root / 'Claude settings'
        self.codex_skill = self.home / '.agents/skills' / SKILL
        self.claude_skill = self.claude_dir / 'skills' / SKILL
        env = {'CLAUDE_CONFIG_DIR': str(self.claude_dir), 'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR': ''}
        self.enterContext(patch.dict(os.environ, env))
        self.enterContext(patch.object(Path, 'home', return_value=self.home))

    def install(self, *args):
        self.assertTrue(INSTALLER.is_file(), 'The user-level skill installer is missing')
        errors = StringIO()
        with patch.object(sys, 'argv', [str(INSTALLER), *args]), redirect_stdout(StringIO()), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(INSTALLER), run_name='__main__')
        return result.exception.code, errors.getvalue()

    def test_installs_both_clients_with_an_absolute_runtime(self):
        self.assertEqual(self.install(), (0, ''))
        for directory in (self.codex_skill, self.claude_skill):
            self.assertEqual((directory / 'SKILL.md').read_bytes(), SOURCE.read_bytes())
            data = json.loads((directory / 'gaid-installation.json').read_text(encoding='utf-8'))
            self.assertEqual(Path(data['project_root']), PROJECT)
            self.assertEqual(Path(data['python_executable']), Path(sys.executable))
            self.assertEqual(Path(data['model_dir']), PROJECT / 'models')

    def test_reinstall_refreshes_external_model_location_for_one_client(self):
        self.assertEqual(self.install('--agent', 'claude'), (0, ''))
        models = self.root / '外部 模型'
        with patch.dict(os.environ, {'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR': str(models)}):
            self.assertEqual(self.install('--agent', 'claude'), (0, ''))
        record = self.claude_skill / 'gaid-installation.json'
        self.assertEqual(Path(json.loads(record.read_text(encoding='utf-8'))['model_dir']), models)
        self.assertFalse(self.codex_skill.exists())

    def test_existing_unmanaged_skill_is_preserved(self):
        self.claude_skill.mkdir(parents=True)
        existing = self.claude_skill / 'SKILL.md'
        existing.write_bytes(b'My existing skill')
        status, errors = self.install()
        self.assertNotEqual(status, 0)
        self.assertIn(str(existing), errors)
        self.assertEqual(existing.read_bytes(), b'My existing skill')
        self.assertFalse(self.codex_skill.exists())
