"""V2 residual restoration, conservative soft weights and protected-pixel blending.

Inputs are RGB sRGB arrays on the same original pixel grid. Residual decomposition
uses the supplied sRGB sample values, not linear radiance. ``blend`` alone converts
active pixels to linear RGB; uint8 quantization occurs once at its output.

Gaussian residual restoration is an experimental frequency split, not a noise
classifier or a guarantee of edge preservation. If the candidate moves an edge,
the residual contains that edge too; semantic and structural vetoes remain useful.
"""

from numbers import Real

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, maximum_filter


def _original_rgb(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise TypeError('original must be a uint8 NumPy RGB array')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('original must have non-empty shape HxWx3')
    return image


def _float_array(value, shape, name):
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.floating):
        raise TypeError(f'{name} must be a floating-point NumPy array')
    if value.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {value.shape}')
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f'{name} must contain finite values in [0, 1]')
    return value.astype(np.float32, copy=False)


def _bool_mask(mask, shape, name):
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise TypeError(f'{name} must be a boolean NumPy array')
    if mask.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {mask.shape}')
    return mask


def _radius(value, name, strictly_positive=False):
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be a real number')
    value = float(value)
    if not np.isfinite(value) or value < 0 or (strictly_positive and value == 0):
        requirement = 'positive' if strictly_positive else 'non-negative'
        raise ValueError(f'{name} must be finite and {requirement}')
    return value


def validate_unit_interval(value, name):
    """Normalize a finite real strength in [0, 1], rejecting booleans."""
    if (not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            or not 0 <= value <= 1):
        raise ValueError(f'{name} must be a finite real number in [0, 1]')
    return float(value)


def gradient_preserving_candidate(original, denoised, sigma, *, restoration_gain=1):
    """Return float32 ``D + restoration_gain * Gaussian(I-D, sigma)`` clipped to [0, 1].

    ``original`` is uint8 HxWx3. ``denoised`` is floating-point HxWx3 in
    [0, 1]; its values are interpreted in the same sRGB encoding as original.
    Sigma is a positive spatial radius in original-image pixels. RGB channels
    are filtered independently with reflected image boundaries. The gain is
    in [0, 1], with zero returning a copy of the uncorrected candidate.

    The correction restores broad differences erased by a denoiser. Broad noise
    can also return, and an already displaced edge can produce halos. The caller
    must evaluate those trade-offs and limit application with a reviewed mask.
    """
    original = _original_rgb(original)
    denoised = _float_array(denoised, original.shape, 'denoised')
    sigma = _radius(sigma, 'sigma', strictly_positive=True)
    restoration_gain = validate_unit_interval(restoration_gain, 'restoration gain')
    if restoration_gain == 0:
        return denoised.copy()
    residual = original.astype(np.float32) / 255 - denoised
    correction = gaussian_filter(residual, sigma=(sigma, sigma, 0), mode='reflect')
    if restoration_gain != 1:
        correction *= np.float32(restoration_gain)
    return np.clip(denoised + correction, 0, 1).astype(np.float32, copy=False)


def _to_linear(value):
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)


def blend(original_uint8, candidate_float, weight_float):
    """Blend in linear RGB; copy W=0 exactly and quantize only the final RGB.

    W=1 uses the candidate directly, avoiding a redundant transfer-function
    round trip at the endpoint. Inputs are validated and never modified.
    """
    original = _original_rgb(original_uint8)
    candidate = _float_array(candidate_float, original.shape, 'candidate')
    weight = _float_array(weight_float, original.shape[:2], 'weight')
    result = original.copy()
    mixed = (weight > 0) & (weight < 1)
    if mixed.any():
        source_linear = _to_linear(original[mixed].astype(np.float32) / 255)
        candidate_linear = _to_linear(candidate[mixed])
        value = source_linear + weight[mixed, None] * (candidate_linear - source_linear)
        result[mixed] = np.rint(_to_srgb(value) * 255).astype(np.uint8)
    full = weight == 1
    if full.any():
        result[full] = np.rint(candidate[full] * 255).astype(np.uint8)
    return result


def _distance_weight(inside, radius):
    if (np.isscalar(radius) and radius == 0) or inside.all() or not inside.any():
        return inside.astype(np.float32)
    distance = distance_transform_edt(inside)
    return (np.sin(np.clip(distance / radius, 0, 1) * np.pi / 2) ** 2).astype(np.float32)


def _gradient_magnitude(gray, sigma):
    smoothed = gaussian_filter(gray, sigma=sigma, mode='reflect')
    gy = np.gradient(smoothed, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    gx = np.gradient(smoothed, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    # Scale-normalize the derivative so an ideal step has comparable response
    # at both smoothing scales. This is a structural heuristic, not confidence.
    return np.hypot(gx, gy) * sigma


def _line_weight(original, structure_radius, gradient_threshold):
    gray = original.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    persistent = (_gradient_magnitude(gray, 1) > gradient_threshold)
    persistent &= _gradient_magnitude(gray, 2) > gradient_threshold
    return _distance_weight(~maximum_filter(persistent, size=3), structure_radius)


def _compose_weight(editable, feather_radius, line_weight, structure_protection):
    weight = _distance_weight(editable, feather_radius)
    if line_weight is not None:
        guarded_weight = weight * line_weight
        if structure_protection == 1:
            weight = guarded_weight
        else:
            weight = (np.float32(1 - structure_protection) * weight
                      + np.float32(structure_protection) * guarded_weight)
    weight[~editable] = 0
    return weight.astype(np.float32, copy=False)


def whole_weight(rgb, alpha, *, strength, structure_protection):
    """Build the whole-image blend weight and its fully transparent protection mask."""
    protected = (alpha == 0 if alpha is not None else np.zeros(rgb.shape[:2], dtype=bool))
    weight = build_weight(~protected, protected, rgb, feather_radius=0,
                          structure_radius=5 * (min(rgb.shape[:2]) / 2048), gradient_threshold=4,
                          structure_protection=structure_protection)
    weight *= np.float32(strength)
    weight[weight < 1 / 255] = 0
    return weight, protected


def build_weight(allowed_bool, protected_bool, original_uint8, feather_radius=16,
                 structure_radius=5, gradient_threshold=4, *, structure_protection=1):
    """Return a float32 soft weight strictly inside allowed and outside protected.

    Main structures are pixels above ``gradient_threshold`` at both Gaussian
    scales 1 and 2, measured in 8-bit luminance units with scale-normalized
    gradients. A 1-pixel expansion covers thin line centers, whose symmetric
    derivative can be zero. Structural distance softening and region feathering
    are inward.
    Feather/structure radii are original-image pixels and must be non-negative.
    Structure protection interpolates between feathering alone (0) and full
    structural softening (1). A zero structure radius also disables it.
    Explicit protected pixels always win, including when both radii are zero.

    This heuristic can still retain some structured noise or miss weak artistic
    lines. Neither the weight nor its area is a learned noise-detection score.
    """
    original = _original_rgb(original_uint8)
    shape = original.shape[:2]
    allowed = _bool_mask(allowed_bool, shape, 'allowed')
    protected = _bool_mask(protected_bool, shape, 'protected')
    feather_radius = _radius(feather_radius, 'feather_radius')
    structure_radius = _radius(structure_radius, 'structure_radius')
    gradient_threshold = _radius(gradient_threshold, 'gradient_threshold')
    structure_protection = validate_unit_interval(structure_protection, 'structure protection')
    editable = allowed & ~protected
    lines = (_line_weight(original, structure_radius, gradient_threshold)
             if structure_protection > 0 and structure_radius > 0 and editable.any() else None)
    return _compose_weight(editable, feather_radius, lines, structure_protection)


def keep_prompt_components(mask_bool, positive_points_xy):
    """Keep every 8-connected component containing at least one positive point.

    Coordinates are finite (x, y) positions in original-image pixels; fractional
    positions refer to their containing pixel (floor). Out-of-image points are
    errors. A point falling on predicted background contributes no component;
    this function never invents foreground to satisfy a prompt. No prompts
    returns an empty selection rather than silently selecting the largest part.
    """
    if not isinstance(mask_bool, np.ndarray) or mask_bool.dtype != np.bool_:
        raise TypeError('mask must be a boolean NumPy array')
    if mask_bool.ndim != 2 or min(mask_bool.shape) == 0:
        raise ValueError('mask must have non-empty shape HxW')
    points = np.asarray(positive_points_xy, dtype=np.float64)
    if points.size == 0:
        return np.zeros_like(mask_bool)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('positive points must be finite Nx2 (x, y) coordinates')
    height, width = mask_bool.shape
    if ((points[:, 0] < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any() or (points[:, 1] >= height).any()):
        raise ValueError('positive points must be inside the image')
    components, _ = label(mask_bool, structure=np.ones((3, 3), dtype=np.uint8))
    pixels = np.floor(points).astype(np.intp)
    selected = np.unique(components[pixels[:, 1], pixels[:, 0]])
    selected = selected[selected != 0]
    return np.isin(components, selected)
