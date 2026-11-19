"""Install the GAID skill into user-level Codex and Claude Code directories."""

import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
from tempfile import mkstemp


SKILL_NAME = 'gpt-animeimage-denoise'
INSTALLATION_FILE = 'gaid-installation.json'


def check_destination(directory, payload):
    if (directory.is_symlink() or directory.is_junction()
            or (directory.exists() and not directory.is_dir())):
        raise FileExistsError(f'Expected a regular skill directory: {directory}')
    skill = directory / 'SKILL.md'
    record = directory / INSTALLATION_FILE
    for path in (skill, record):
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise FileExistsError(f'Expected a regular file: {path}')
    if skill.is_file() and not record.is_file() and skill.read_bytes() != payload:
        raise FileExistsError(f'Existing skill has no GAID installation record: {skill}')


def write_file(path, data):
    descriptor, name = mkstemp(dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=('both', 'codex', 'claude'), default='both',
                        help='Install for one client or both (default: both)')
    args = parser.parse_args(argv)
    if find_spec('gpt_animeimage_denoise') is None:
        parser.error('Install GAID in this Python environment first. See README.md.')

    project = Path(__file__).resolve().parents[1]
    source = project / '.agents' / 'skills' / SKILL_NAME / 'SKILL.md'
    home = Path.home()
    claude_config = Path(os.environ.get('CLAUDE_CONFIG_DIR') or home / '.claude').expanduser()
    roots = {'codex': home / '.agents' / 'skills', 'claude': claude_config / 'skills'}
    selected = roots if args.agent == 'both' else (args.agent,)
    destinations = [roots[client] / SKILL_NAME for client in selected]
    models = Path(os.environ.get('GPT_ANIMEIMAGE_DENOISE_MODEL_DIR') or project / 'models').expanduser().resolve()
    data = {'project_root': str(project), 'python_executable': sys.executable, 'model_dir': str(models)}
    record = (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    try:
        payload = source.read_bytes()
        for directory in destinations:
            check_destination(directory, payload)
        for directory in destinations:
            directory.mkdir(parents=True, exist_ok=True)
            write_file(directory / INSTALLATION_FILE, record)
            write_file(directory / 'SKILL.md', payload)
            print(f'Installed {directory / "SKILL.md"}')
    except OSError as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
