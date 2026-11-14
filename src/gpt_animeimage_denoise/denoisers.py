"""Original-grid candidate stages for the V2.1 fusion pipeline."""
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
from PIL import Image

from .model_paths import model_root
from .pipeline import _l0_callable_signature, _sha

DENOISERS = ('l0', 'esrgan', 'esrgan-l0', 'apisr')
_ALIASES = {'esrgan60_l0': 'esrgan-l0', 'esrgan60': 'esrgan',
            'l0_0.001': 'l0', 'l0_0.0005': 'l0'}
METHOD_CHOICES = (*DENOISERS, *_ALIASES)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESRGAN_MODEL = 'RealESRGAN_x4plus_anime_6B.pth'
ESRGAN_SHA256 = 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
DOWNSCALE = 0.6


def resolve_denoiser(method, lam=None):
    """Resolve upstream names while retaining an explicit L0 strength override."""
    requested = 'l0' if method is None else method
    canonical = _ALIASES.get(requested, requested)
    if canonical not in DENOISERS:
        raise ValueError('Unknown denoiser: ' + str(method))
    strength = (0.001 if requested == 'l0_0.001' else 0.0005) if lam is None else lam
    if not isinstance(strength, Real) or not math.isfinite(strength) or strength <= 0:
        raise ValueError('L0 lambda must be a finite positive value')
    return canonical, float(strength)


def uses_sr(method):
    return resolve_denoiser(method)[0] != 'l0'


def uses_l0(method):
    canonical = resolve_denoiser(method)[0]
    return canonical == 'l0' or canonical.endswith('-l0')


def _sr_settings(method, downscale):
    canonical = resolve_denoiser(method)[0]
    if canonical == 'l0':
        raise ValueError('Pure L0 has no super-resolution stage')
    if (not isinstance(downscale, Real) or not math.isfinite(downscale)
            or not 0 < downscale <= 1):
        raise ValueError('SR downscale must be finite and in 0..1')
    network = canonical.removesuffix('-l0')
    return network, 1.0 if network == 'apisr' else float(downscale)


def _model_spec(network):
    import denoise
    name = denoise.MODELS[network][0]
    # Preserve the published ESRGAN constant for existing callers/adapters.
    checksum = ESRGAN_SHA256 if network == 'esrgan' else denoise.SHA256[name]
    return name, checksum


def describe_denoiser(method, device, *, sr_fn=None, downscale=DOWNSCALE):
    """Return a JSON identity without loading models or downloading weights."""
    canonical = resolve_denoiser(method)[0]
    identity = {'method': canonical, 'device': str(device)}
    if canonical == 'l0':
        return identity
    network, effective_scale = _sr_settings(canonical, downscale)
    identity.update({'network': network, 'downscale': effective_scale,
        'precision': {'dtype': 'float32', 'autocast': False,
                      'matmul_allow_tf32': False, 'cudnn_allow_tf32': False},
        'implementation': {name: _sha(path) for name, path in [
            ('denoisers.py', Path(__file__).resolve()),
            ('denoise.py', PROJECT_ROOT / 'denoise.py'),
            ('tiled_inference.py', PROJECT_ROOT / 'tiled_inference.py')]}})
    if sr_fn is None:
        name, checksum = _model_spec(network)
        identity['model'] = {'name': name,
            'path': str(model_root() / name), 'expected_sha256': checksum}
        identity['sr_implementation'] = {'kind': 'installed_' + network}
    else:
        if not callable(sr_fn):
            raise ValueError('sr_fn must be callable')
        try:
            signature = _l0_callable_signature(sr_fn)
        except ValueError as error:
            raise ValueError(str(error).replace('l0_fn', 'sr_fn')) from error
        signature['kind'] = 'injected_sr'
        identity['sr_implementation'] = signature
    # Normalize tuple defaults/signatures so saved JSON metadata compares equally.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def _validate_rgb(rgb, *, shape=None):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1
            or (shape is not None and rgb.shape != shape)):
        raise ValueError('RGB must be a nonempty uint8 HxWx3 array on the original grid')


def _native_super_resolve(rgb, device, network, downscale):
    import denoise
    import torch

    # Legacy fetch validates downloads but trusts existing files; validate both
    # paths before load_model can deserialize any model data.
    _, checksum = _model_spec(network)
    assets = model_root()
    path = Path(denoise.fetch(network, assets))
    if _sha(path) != checksum:
        raise ValueError(network.upper() + ' SHA256 checksum mismatch: ' + str(path))
    image = Image.fromarray(rgb)
    width, height = image.size
    small = image if network == 'apisr' else image.resize(
        (max(1, round(width * downscale)), max(1, round(height * downscale))),
        Image.Resampling.BOX)
    device_type = torch.device(device).type
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    model = None
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type=device_type, enabled=False), torch.inference_mode():
            model = denoise.load_model(network, assets, device).float().eval()
            # Keep the established shared 256-core/64-context/64-overlap tiler.
            enlarged = denoise.upscale(model, small, device)
            return np.array(enlarged.resize((width, height), Image.Resampling.LANCZOS),
                            dtype=np.uint8, copy=True)
    finally:
        model = None
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
        if device_type == 'cuda':
            torch.cuda.empty_cache()


def super_resolve(rgb_uint8, *, device, sr_fn=None, method='esrgan-l0', downscale=DOWNSCALE):
    """Run a selected network and return quantized RGB on the original grid."""
    _validate_rgb(rgb_uint8)
    network, effective_scale = _sr_settings(method, downscale)
    if sr_fn is not None and not callable(sr_fn):
        raise ValueError('sr_fn must be callable')
    result = (_native_super_resolve(rgb_uint8, device, network, effective_scale) if sr_fn is None else
              sr_fn(rgb_uint8.copy(), device=device, downscale=effective_scale))
    _validate_rgb(result, shape=rgb_uint8.shape)
    return result


def smooth_candidate(rgb_uint8, lam, *, device, l0_fn=None):
    """Run L0 on uint8 RGB without adding low-frequency restoration or blending."""
    _validate_rgb(rgb_uint8)
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError('L0 lambda must be a finite positive value')
    if l0_fn is None:
        from denoise import l0_smooth
        l0_fn = l0_smooth
    elif not callable(l0_fn):
        raise ValueError('l0_fn must be callable')
    result = np.asarray(l0_fn(rgb_uint8.astype(np.float64) / 255, float(lam), device=device))
    if (result.shape != rgb_uint8.shape or not np.isfinite(result).all()
            or np.any((result < 0) | (result > 1))):
        raise ValueError('L0 candidate must be finite HxWx3 RGB in 0..1 on the original grid')
    return result.astype(np.float32)
