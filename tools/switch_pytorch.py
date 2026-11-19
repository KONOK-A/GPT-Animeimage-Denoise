"""Switch GAID's virtual environment between the CPU and CUDA PyTorch builds."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib


BUILDS = {'cpu': ('cpu', None), 'cuda': ('cu128', '12.8')}
PACKAGES = ('torch', 'torchvision')


def _target_versions(tag):
    path = Path(__file__).resolve().parents[1] / 'pyproject.toml'
    dependencies = tomllib.loads(path.read_text(encoding='utf-8'))['project']['dependencies']
    pins = {}
    for requirement in dependencies:
        name, operator, version = requirement.partition('==')
        if name in PACKAGES and operator:
            pins[name] = version
    if len(pins) != len(PACKAGES):
        raise ValueError('pyproject.toml must pin torch and torchvision before switching builds.')
    return {name: f'{pins[name]}+{tag}' for name in PACKAGES}


def _pip(*arguments):
    subprocess.run([sys.executable, '-I', '-X', 'utf8', '-m', 'pip', '--disable-pip-version-check', *arguments],
                   check=True)


def _runtime():
    probe = (
        'import json, torch, torchvision\n'
        'print(json.dumps(dict(torch=torch.__version__, torchvision=torchvision.__version__, '
        'cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())))'
    )
    result = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', probe],
                            check=True, capture_output=True, encoding='utf-8')
    return json.loads(result.stdout.strip().rsplit('\n', 1)[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=BUILDS, help='cpu or NVIDIA CUDA 12.8')
    args = parser.parse_args(argv)
    if sys.prefix == sys.base_prefix:
        parser.error('Use the Python interpreter in the GAID virtual environment.')

    tag, expected_cuda = BUILDS[args.target]
    try:
        expected = _target_versions(tag)
        requirements = [f'{name}=={version}' for name, version in expected.items()]
        index = f'https://download.pytorch.org/whl/{tag}'
        with TemporaryDirectory(prefix='gaid-pytorch-', dir=Path(sys.prefix).resolve()) as wheels:
            print(f'Downloading the {tag} builds of torch and torchvision.', flush=True)
            _pip('download', '--only-binary=:all:', '--no-deps', '--no-cache-dir',
                 '--index-url', index, '--dest', wheels, *requirements)
            print('Both wheels downloaded. Replacing PyTorch and torchvision offline.', flush=True)
            _pip('install', '--no-index', '--find-links', wheels, '--force-reinstall', '--no-deps', *requirements)

        runtime = _runtime()
        build = 'CPU' if runtime['cuda'] is None else f'CUDA {runtime["cuda"]}'
        actual_device = 'CUDA' if runtime['cuda_available'] else 'CPU'
        print(f'PyTorch {runtime["torch"]} · torchvision {runtime["torchvision"]} · {build} build')
        print(f'GAID auto device is {actual_device}.')
        if any(runtime[name] != version for name, version in expected.items()) or runtime['cuda'] != expected_cuda:
            raise RuntimeError(f'The imported PyTorch build does not match the requested {tag} wheels.')
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.rstrip(), file=sys.stderr)
        print(f'PyTorch switch failed with exit code {error.returncode}.', file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
