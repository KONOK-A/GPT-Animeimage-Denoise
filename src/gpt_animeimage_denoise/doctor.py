"""Inspect the local GAID installation without changing it."""
import argparse
from contextlib import redirect_stdout
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import platform
import re
from shlex import quote
import sys
from urllib.parse import urlsplit
from urllib.request import url2pathname

if not __package__:
    # The checkout entry point also runs before an editable install exists.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpt_animeimage_denoise import __version__
from gpt_animeimage_denoise.devices import DEVICE_HELP, resolve_device
from gpt_animeimage_denoise.model_paths import MODEL_ENV, model_root


PROJECT = 'gpt-animeimage-denoise'
MODEL_MANIFEST = Path(__file__).with_name('model_manifest.json')
SOURCE_ROOT = Path(__file__).resolve().parents[2]
_EMPTY_SHA256 = hashlib.sha256(b'').hexdigest()


def dependency_problems():
    """Read install metadata without importing any image or inference libraries."""
    problems = []
    if sys.version_info < (3, 12):
        problems.append('当前 Python 低于 3.12。')
    try:
        requirements = metadata.requires(PROJECT) or ()
    except metadata.PackageNotFoundError:
        requirements = ()
        problems.append(f'缺少依赖包 {PROJECT}。')

    try:
        from packaging.requirements import Requirement
    except ImportError:
        Requirement = None

    missing = []
    for requirement in requirements:
        spec = Requirement(requirement) if Requirement is not None else None
        if spec is not None and spec.marker is not None and not spec.marker.evaluate():
            continue
        name = spec.name if spec is not None else re.split(r'[<>=!~;]', requirement, maxsplit=1)[0].strip()
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if spec is not None and spec.specifier and installed not in spec.specifier:
            problems.append(f'{name} 已安装 {installed}，项目要求 {spec.specifier}。')
    if missing:
        problems.append(f'缺少依赖包 {", ".join(missing)}。')
    return problems


def _project_command(arguments):
    python = r'& .\.venv\Scripts\python.exe' if os.name == 'nt' else './.venv/bin/python'
    return f'{python} {arguments}'


def _issue(issues, code, message, repair, *, level='error'):
    issues.append({'level': level, 'code': code, 'message': message, 'repair': repair})


def _editable_source(issues):
    try:
        distribution = metadata.distribution(PROJECT)
        raw = distribution.read_text('direct_url.json')
    except (metadata.PackageNotFoundError, OSError):
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
        if not value.get('dir_info', {}).get('editable'):
            return None
        url = urlsplit(value['url'])
        if url.scheme != 'file':
            return None
        original = Path(url2pathname(('//' + url.netloc if url.netloc else '') + url.path)).resolve()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    if not (original / 'pyproject.toml').is_file() or (
            (SOURCE_ROOT / 'pyproject.toml').is_file() and original != SOURCE_ROOT):
        _issue(issues, 'editable_install', f'可编辑安装仍指向 {original}。',
               '在当前项目目录重建虚拟环境并按 README.md 安装，再运行 '
               + _project_command('-m pip install -e .'))
    return str(original)


def _skill_record(path, issues):
    repair = '在新项目目录重建或启用 GAID 环境，再运行 ' + _project_command('tools/install_agent_skill.py')
    record = {'file': str(path.resolve())}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        for key in ('project_root', 'python_executable', 'model_dir'):
            value = Path(data[key])
            if not value.is_absolute():
                raise ValueError(f'{key} 需要绝对路径')
            record[key] = str(value)
    except (OSError, ValueError, KeyError, TypeError) as error:
        _issue(issues, 'skill_installation', f'无法读取 Skill 安装记录。{error}', repair)
        return record

    project = Path(record['project_root'])
    python = Path(record['python_executable'])
    if not (project / 'pyproject.toml').is_file() or not python.is_file():
        _issue(issues, 'skill_installation', 'Skill 记录中的项目或解释器路径已失效。', repair)
    elif os.path.normcase(str(python)) != os.path.normcase(sys.executable):
        _issue(issues, 'skill_installation', '当前解释器与 Skill 安装记录不同。',
               f'使用记录里的解释器 {python} 再测一次。若已更换环境，' + repair, level='warning')
    return record


def _check_image_runtime(issues):
    for package, module in (('numpy', 'numpy'), ('Pillow', 'PIL.Image'),
                            ('scipy', 'scipy.ndimage'), ('opencv-python', 'cv2')):
        try:
            import_module(module)
        except Exception as error:
            _issue(issues, 'image_runtime', f'无法导入 {package}。{type(error).__name__} {error}',
                   '在当前 GAID 环境重新安装该包，或按 README.md 重建虚拟环境。')


def _repair_torch():
    install = _project_command('-m pip install -e .')
    return f'按 README.md 或 docs/accelerators.md 安装配对的 PyTorch 和 torchvision，然后运行 {install}'


def _runtime(requested, issues):
    pytorch = {'torch': None, 'torchvision': None, 'build': None, 'cuda': None, 'hip': None}
    device = {'requested': requested, 'selected': None, 'name': None, 'tested': False}
    try:
        import torch
    except Exception as error:
        _issue(issues, 'torch', f'无法导入 PyTorch。{type(error).__name__} {error}', _repair_torch())
        return pytorch, device

    cuda_version = torch.version.cuda
    hip_version = torch.version.hip
    build = 'rocm' if hip_version else 'cuda' if cuda_version else (
        'mps' if torch.backends.mps.is_built() else 'cpu')
    pytorch.update(torch=str(torch.__version__), cuda=cuda_version, hip=hip_version, build=build)
    try:
        import torchvision
        pytorch['torchvision'] = str(torchvision.__version__)
    except Exception as error:
        _issue(issues, 'torchvision', f'无法导入 torchvision。{type(error).__name__} {error}', _repair_torch())

    repair = '请检查驱动和 PyTorch 构建是否匹配，或在命令中指定 --device cpu。参见 docs/accelerators.md。'
    if os.name == 'nt' and requested.startswith('cuda') and build == 'cpu':
        repair = ('NVIDIA 用户可在项目目录运行 ' + _project_command('tools/switch_pytorch.py cuda')
                  + '。AMD 用户请按 docs/accelerators.md 安装 ROCm。也可指定 --device cpu。')
    try:
        selected = resolve_device(requested)
        device['selected'] = selected
        with torch.inference_mode():
            sample = torch.ones((2, 2), device=selected)
            if sample.sum().item() != 4:
                raise RuntimeError('推理设备返回了异常结果')
        if selected.startswith('cuda'):
            device['name'] = torch.cuda.get_device_name(selected)
        elif selected == 'mps':
            device['name'] = 'Apple Metal'
        else:
            device['name'] = 'CPU'
        device['tested'] = True
    except Exception as error:
        _issue(issues, 'device', f'推理设备检查失败。{type(error).__name__} {error}', repair)
    if requested == 'auto' and device['tested'] and device['selected'] == 'cpu' and build != 'cpu':
        _issue(issues, 'device_fallback', 'PyTorch 带 GPU 支持，但本次自动选择了 CPU。', repair, level='warning')
    return pytorch, device


def _path_argument(path):
    text = str(path)
    return "'" + text.replace("'", "''") + "'" if os.name == 'nt' else quote(text)


def _models(root, source, whole, verify_models, issues):
    report = {'root': str(root), 'source': source, 'scope': 'whole' if whole else 'parts',
              'verified': verify_models, 'checked': 0, 'missing': [], 'invalid': [],
              'optional_missing': [], 'optional_invalid': []}
    try:
        manifest = json.loads(MODEL_MANIFEST.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        _issue(issues, 'model_catalog', f'无法读取 GAID 模型清单。{error}',
               '请恢复项目安装，再重试诊断。')
        return report

    for name, entry in manifest.items():
        optional = entry.get('optional', False)
        if whole and not optional:
            continue
        path = root / name
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError('模型文件指向目录外部')
            if not path.is_file():
                key = 'invalid' if path.exists() or path.is_symlink() else 'missing'
            else:
                if path.stat().st_size == 0 and entry['sha256'] != _EMPTY_SHA256:
                    raise ValueError('空文件')
                if verify_models:
                    with path.open('rb') as stream:
                        if hashlib.file_digest(stream, 'sha256').hexdigest() != entry['sha256']:
                            raise ValueError('SHA-256 不匹配')
                report['checked'] += 1
                continue
        except (OSError, ValueError, RuntimeError):
            key = 'invalid'
        report[('optional_' if optional else '') + key].append(name)

    fetch = _project_command('-m tools.assets fetch --dest ' + _path_argument(root))
    if report['missing']:
        _issue(issues, 'models_missing', f'缺少 {len(report["missing"])} 个分区资产。',
               f'在项目目录运行 {fetch}')
    if report['invalid']:
        _issue(issues, 'models_invalid', f'发现 {len(report["invalid"])} 个异常分区资产。',
               f'先备份并移出报告列出的异常文件，再在项目目录运行 {fetch}')
    if report['optional_invalid']:
        _issue(issues, 'optional_models_invalid', '检测到异常的可选超分权重。',
               '先备份并移出对应权重。再次选用该算法时，GAID 会下载缺少的权重。', level='warning')
    return report


def diagnose(*, device='auto', whole=False, verify_models=False, model_dir=None, installation=None):
    """Return a shared diagnosis report without installing anything or loading model weights."""
    issues = []
    problems = dependency_problems()
    if problems:
        _issue(issues, 'dependencies', ' '.join(problems),
               '按 README.md 准备 Python 3.12 或更新版本及 PyTorch，再在项目目录运行 '
               + _project_command('-m pip install -e .'))
    editable = _editable_source(issues)
    record = _skill_record(installation, issues) if installation is not None else None
    if model_dir is not None:
        root, source = model_dir.expanduser().resolve(), 'override'
    elif record and record.get('model_dir'):
        root, source = Path(record['model_dir']).resolve(), 'installation'
    else:
        root, source = model_root(), 'environment' if os.environ.get(MODEL_ENV) else 'cwd'
    _check_image_runtime(issues)
    pytorch, selected = _runtime(device, issues)
    models = _models(root, source, whole, verify_models, issues)
    report = {'status': 'error' if any(item['level'] == 'error' for item in issues) else 'ok',
              'version': __version__,
              'python': {'executable': sys.executable, 'version': platform.python_version(), 'prefix': sys.prefix},
              'package': {'source': str(Path(__file__).resolve().parent), 'editable_root': editable},
              'pytorch': pytorch, 'device': selected, 'models': models, 'issues': issues}
    if record is not None:
        report['installation'] = record
    return report


def format_report(report):
    python, pytorch, device, models = (report[key] for key in ('python', 'pytorch', 'device', 'models'))
    build = {
        'cuda': f'CUDA {pytorch["cuda"]}', 'rocm': f'ROCm {pytorch["hip"]}', 'mps': 'MPS', 'cpu': 'CPU',
    }.get(pytorch['build'], '未知构建')
    selected = device['selected'] or '不可用'
    if device['name'] and device['name'] != selected:
        selected += f' · {device["name"]}'
    probe = '算子测试通过' if device['tested'] else '算子测试未通过'
    source = {'override': '命令行指定', 'installation': 'Skill 安装记录',
              'environment': MODEL_ENV, 'cwd': '当前目录的 models'}
    lines = [
        f'GAID {report["version"]} · ' + ('诊断通过' if report['status'] == 'ok' else '发现问题'),
        f'Python {python["version"]} · {python["executable"]}',
        f'GAID 模块  {report["package"]["source"]}',
        f'PyTorch {pytorch["torch"] or "不可用"} · {build}',
        f'torchvision {pytorch["torchvision"] or "不可用"}',
        f'推理设备  {device["requested"]} → {selected} · {probe}',
        f'模型目录  {models["root"]}（{source[models["source"]]}）',
        ('整图纯 L0 不需要分区权重。' if models['scope'] == 'whole' else
         '分区模型  ' + ('已核对 SHA-256' if models['verified'] else '已检查文件存在性和异常的空文件')),
    ]
    if 'installation' in report:
        installation = report['installation']
        lines.append('Skill 记录  ' + installation['file'])
        for key, label in [('project_root', '项目目录'), ('python_executable', '解释器'), ('model_dir', '模型目录')]:
            if key in installation:
                lines.append(f'  {label}  {installation[key]}')
    for key, label in [('missing', '缺少'), ('invalid', '异常'), ('optional_missing', '可选，首次选用时下载'),
                       ('optional_invalid', '可选权重异常')]:
        if models[key]:
            lines.extend([label, *('  ' + name for name in models[key])])
    for issue in report['issues']:
        lines.extend(['', ('错误  ' if issue['level'] == 'error' else '提示  ') + issue['message'],
                      '  ' + issue['repair']])
    return '\n'.join(lines)


def add_arguments(parser):
    parser.add_argument('--device', default='auto', help=DEVICE_HELP)
    parser.add_argument('--whole', action='store_true', help='Check whole-image L0 without requiring part models')
    parser.add_argument('--model-dir', type=Path, help='Override the model root for this check')
    parser.add_argument('--verify-models', action='store_true', help='Read model files and verify SHA-256')
    parser.add_argument('--installation', type=Path, help='Check a user-level Skill installation JSON')
    parser.add_argument('--json', action='store_true', help='Print a single machine-readable JSON report')


def run_command(args):
    with redirect_stdout(sys.stderr):
        report = diagnose(device=args.device, whole=args.whole, verify_models=args.verify_models,
                          model_dir=args.model_dir, installation=args.installation)
    output = json.dumps(report, ensure_ascii=True) if args.json else format_report(report)
    encoding = getattr(sys.stdout, 'encoding', None)
    if not args.json and encoding:
        output = output.encode(encoding, errors='backslashreplace').decode(encoding)
    print(output)
    return int(report['status'] == 'error')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_command(parser.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
