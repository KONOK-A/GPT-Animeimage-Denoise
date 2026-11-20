"""Denoiser names and parameter rules shared by inference and the CLI."""
import math
from numbers import Real


DENOISERS = ('l0', 'esrgan', 'esrgan-l0', 'apisr')
_ALIASES = {'esrgan60_l0': 'esrgan-l0', 'esrgan60': 'esrgan',
            'l0_0.001': 'l0', 'l0_0.0005': 'l0'}
METHOD_CHOICES = (*DENOISERS, *_ALIASES)
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
