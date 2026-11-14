"""Editable selection and exact compositing on the original image grid."""
import math

import numpy as np
from PIL import Image, ImageDraw

from .corrections import MODULES as PERSON_MODULES
from .filters import (_bool_mask, _original_rgb, _radius, blend, build_weight,
                      gradient_preserving_candidate, validate_unit_interval)

MODULES = {'background': 0, **PERSON_MODULES}
DEFAULT_MODULES = ('skin', 'face', 'hair')


class SelectionEditor:
    """Combine module choices and manual strokes under immutable vetoes."""

    def __init__(self, rgb, alpha, labels, objects, unknown):
        self.rgb = _original_rgb(rgb)
        shape = rgb.shape[:2]
        if not isinstance(labels, np.ndarray) or labels.dtype != np.uint8 or labels.shape != shape:
            raise ValueError('Labels must be original-grid uint8')
        if not np.isin(labels, (*MODULES.values(), 255)).all():
            raise ValueError('Labels must be in 0..6 or 255')
        if alpha is not None and (not isinstance(alpha, np.ndarray) or alpha.dtype != np.uint8
                                  or alpha.shape != shape):
            raise ValueError('Alpha must be original-grid uint8 or None')
        self.alpha = alpha
        self.labels = labels
        self.objects = _bool_mask(objects, shape, 'objects').copy()
        self.unknown = _bool_mask(unknown, shape, 'unknown') | (labels == 255)
        self.objects.setflags(write=False)
        self.unknown.setflags(write=False)
        self.hard_protection = self.objects | self.unknown
        if alpha is not None:
            self.hard_protection |= alpha == 0
        self.hard_protection.setflags(write=False)
        self._strokes = np.zeros(shape, np.int8)
        self._scale = min(shape) / 2048

    def paint_stroke(self, points_xy, radius, *, erase=False):
        """Set a round stroke in original pixel coordinates. The last stroke wins."""
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or not np.isfinite(points).all():
            raise ValueError('Stroke points must be a nonempty list of finite [x, y] pairs')
        h, w = self.labels.shape
        if (points < 0).any() or (points[:, 0] >= w).any() or (points[:, 1] >= h).any():
            raise ValueError('Stroke points must lie within the image')
        radius = min(math.hypot(w, h), _radius(radius, 'brush radius', strictly_positive=True))
        radius = max(1, round(radius))
        pixels = np.floor(points).astype(np.intp)
        x0, y0 = np.maximum(0, pixels.min(axis=0) - radius)
        x1, y1 = np.minimum((w, h), pixels.max(axis=0) + radius + 1)
        mask = Image.new('1', (int(x1 - x0), int(y1 - y0)))
        draw = ImageDraw.Draw(mask)
        local = [tuple(map(int, point)) for point in pixels - (x0, y0)]
        if len(local) > 1:
            draw.line(local, fill=1, width=2 * radius + 1, joint='curve')
        for x, y in local:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=1)
        hits = np.asarray(mask) & ~self.hard_protection[y0:y1, x0:x1]
        strokes = self._strokes[y0:y1, x0:x1]
        strokes[hits] = -1 if erase else 1

    @property
    def erased(self):
        """Return the editable pixels currently excluded by the eraser."""
        return self._strokes < 0

    def clear_strokes(self):
        self._strokes.fill(0)

    def selection(self, modules):
        try:
            labels = [MODULES[name] for name in modules]
        except (KeyError, TypeError) as error:
            raise ValueError('Expected known selection modules') from error
        chosen = np.isin(self.labels, labels)
        return (chosen | (self._strokes > 0)) & (self._strokes >= 0) & ~self.hard_protection

    def weights(self, modules, *, strength=1, structure_protection=.2):
        strength = validate_unit_interval(strength, 'strength')
        weight = build_weight(self.selection(modules), self.hard_protection, self.rgb,
                              feather_radius=16 * self._scale, structure_radius=5 * self._scale,
                              structure_protection=structure_protection)
        weight *= np.float32(strength)
        weight[weight < 1 / 255] = 0
        return weight

    def render(self, candidate, modules, *, strength=1, structure_protection=.2,
               restoration_gain=.25, sigma=12):
        weight = self.weights(modules, strength=strength, structure_protection=structure_protection)
        effective_sigma = _radius(sigma, 'sigma', strictly_positive=True) * self._scale
        restored = gradient_preserving_candidate(self.rgb, candidate, effective_sigma,
                                                 restoration_gain=restoration_gain)
        return blend(self.rgb, restored, weight), weight

    def correction_masks(self, modules):
        """Return allow/protect edits relative to the automatic default scope."""
        default = np.isin(self.labels, [MODULES[name] for name in DEFAULT_MODULES]) & ~self.hard_protection
        chosen = self.selection(modules)
        return chosen & ~default, default & ~chosen
