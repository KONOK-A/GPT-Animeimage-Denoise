"""Shared source-image I/O, path guards, and cache identity helpers."""
import hashlib
import json
import marshal
from pathlib import Path, PureWindowsPath

import numpy as np
from PIL import Image, ImageOps


def _assert_inputs_safe(targets, inputs):
    """Reject path, symlink and hardlink aliases before writing any input file."""
    inputs = [Path(path).resolve() for path in inputs]
    for target in targets:
        target = Path(target)
        resolved = target.resolve()
        for source in inputs:
            if resolved == source or (target.exists() and source.exists() and target.samefile(source)):
                raise ValueError('Output would overwrite an input file or its alias: ' + str(source))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def _l0_callable_signature(function):
    """Identify injected implementations; stateful adapters should set .signature.

    Python code, defaults and JSON-serializable closure values are tracked. An
    explicit stable .signature identifies result-affecting object/external state
    that cannot be inferred from bytecode. Opaque callables require that field.
    """
    if function is None:return {'kind': 'installed_l0'}
    target = function if hasattr(function, '__code__') else getattr(function, '__call__', function)
    code = getattr(target, '__code__', None)
    identity = {'kind': 'injected_l0',
                'code_sha256': hashlib.sha256(marshal.dumps(code)).hexdigest() if code is not None else None}
    explicit = getattr(function, 'signature', None)
    if explicit is not None:
        identity['explicit_signature'] = explicit
    else:
        if code is None:raise ValueError('Injected non-Python l0_fn requires an explicit .signature')
        identity['defaults'] = getattr(target, '__defaults__', None)
        identity['keyword_defaults'] = getattr(target, '__kwdefaults__', None)
        closure = getattr(target, '__closure__', None)
        identity['closure'] = [cell.cell_contents for cell in closure] if closure else []
    try:
        json.dumps(identity, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('Injected l0_fn state requires a stable JSON-serializable .signature') from error
    return identity


def _case_folder(output, case_id):
    case_id = str(case_id)
    if (not case_id or case_id in ['.', '..'] or '/' in case_id or '\\' in case_id
            or PureWindowsPath(case_id).drive or any(c in case_id for c in ':*?"<>|')):
        raise ValueError('case_id must be a single safe folder name')
    folder = (output / case_id).resolve()
    if not folder.is_relative_to(output) or folder == output:
        raise ValueError('Case output must remain under output root')
    return folder


def _read_source(source):
    with Image.open(source) as stored:
        image = ImageOps.exif_transpose(stored)
        image.load()
        icc = image.info.get('icc_profile')
        has_alpha = image.mode in ['RGBA', 'LA'] or 'transparency' in image.info
        rgba = image.convert('RGBA') if has_alpha else None
        rgb = np.array(image.convert('RGB'), dtype=np.uint8, copy=True)
        alpha = np.array(rgba.getchannel('A'), copy=True) if rgba is not None else None
    return rgb, alpha, icc


def _save_rgb(path, rgb, alpha, icc):
    image = Image.fromarray(rgb)
    if alpha is not None:image.putalpha(Image.fromarray(alpha))
    kwargs = {'icc_profile': icc} if icc else {}
    image.save(path, format='PNG', **kwargs)
