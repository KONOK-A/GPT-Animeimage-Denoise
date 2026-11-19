"""Recombine a completed case on the CPU using its exact saved candidate."""
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .filters import _radius, blend, gradient_preserving_candidate, validate_unit_interval
from .fusion_evaluation import _review_candidate
from .pipeline import _assert_inputs_safe, _read_source, _save_rgb


def _verified_output(case, report, name):
    content = (case / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != report['output_hashes'].get(name):
        raise ValueError(f'{name} does not match the completed case report')
    return BytesIO(content)


def preview_case(case_dir, output, *, strength=None, restoration_gain=None, sigma=None):
    """Write a new full-size preview with the source case's fixed mask and denoiser."""
    case = Path(case_dir).resolve(strict=True)
    report_path = case / 'report.json'
    cache = (case.parent / '.cache').resolve()
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'complete' or report.get('profile') not in ('fusion', 'reference', 'whole'):
        raise ValueError('Preview requires a completed GAID case directory')
    if not report.get('candidate_cache_identity') or not report.get('candidate_cache_stage'):
        raise ValueError('Preview requires the exact float candidate cache; run process again for this source')

    destination = Path(output)
    if destination.suffix.lower() != '.png':
        raise ValueError('Preview output must be a PNG path')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Preview output already exists: ' + str(destination))
    destination = destination.resolve()
    if destination.is_relative_to(case) or destination.is_relative_to(cache):
        raise ValueError('Write the preview outside the completed case and its cache')
    _assert_inputs_safe([destination], [report_path, report['source'], *report.get('input_paths', [])])

    parameters = report['parameters']
    saved_strength = validate_unit_interval(parameters['strength'], 'saved strength')
    strength = saved_strength if strength is None else validate_unit_interval(strength, 'strength')
    if saved_strength != 1 and strength not in (0, saved_strength):
        raise ValueError('Run process with --strength 1 to freely change preview strength')
    gain = parameters.get('restoration_gain', 1) if restoration_gain is None else restoration_gain
    gain = validate_unit_interval(gain, 'restoration gain')

    rgb, alpha, icc = _read_source(_verified_output(case, report, 'original.png'))
    shape = rgb.shape[:2]
    if report['size'] != [shape[1], shape[0]]:
        raise ValueError('Case dimensions do not match report')
    if sigma is None:
        effective_sigma = _radius(report['effective_sigma'], 'effective sigma', strictly_positive=True)
    else:
        effective_sigma = _radius(sigma, 'sigma', strictly_positive=True) * (min(shape) / 2048)
    weight = np.load(_verified_output(case, report, 'weight.npy'), allow_pickle=False)
    if (not isinstance(weight, np.ndarray) or weight.dtype != np.float32 or weight.shape != shape or not np.isfinite(weight).all()
            or np.any((weight < 0) | (weight > 1))):
        raise ValueError('Invalid original-grid float32 weight.npy')
    with Image.open(_verified_output(case, report, 'protected.png')) as mask:
        protected = np.array(mask.convert('L')) > 0
    if protected.shape != shape or weight[protected].any() or (alpha is not None and weight[alpha == 0].any()):
        raise ValueError('Saved weight extends into protected or fully transparent pixels')

    candidate = _review_candidate(cache, report, rgb.shape)
    if strength != saved_strength:
        # At saved strength 1, thresholded zeros cannot reactivate at a lower strength.
        weight *= np.float32(strength)
        weight[weight < 1 / 255] = 0
    restored = gradient_preserving_candidate(rgb, candidate, effective_sigma, restoration_gain=gain)
    result = blend(rgb, restored, weight)
    if not np.array_equal(result[weight == 0], rgb[weight == 0]):
        raise AssertionError('Preview changed zero-weight pixels')

    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = destination.open('xb')
    complete = False
    try:
        with stream:
            _save_rgb(stream, result, alpha, icc)
        complete = True
    finally:
        if not complete:
            destination.unlink(missing_ok=True)
    return {'id': report['id'], 'status': 'complete', 'output': str(destination),
            'strength': strength, 'restoration_gain': gain, 'effective_sigma': effective_sigma}
