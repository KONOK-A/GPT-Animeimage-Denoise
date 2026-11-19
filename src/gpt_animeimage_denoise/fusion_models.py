"""Strict, serial model adapters for V2.1 evidence and SAM proposals.

The upstream AnimeSeg implementation is loaded from the pinned production
snapshot. Its permissive checkpoint loader and experiment scripts are never used.
All public coordinates refer to the EXIF-oriented input RGB pixel grid.
"""
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from .model_paths import model_root

PINNED_ANIME = {
    'models/anime_seg_mask2former_v4.safetensors': 'b7b684be5f682e6fd476d60fe4b23a9f3925f43ffb26ff8383220b5117954ea1',
    'base-configs/large/config.json': '0f01771fc9adae7f47fd27078e8c800afd655b1b752969f866e5e2b3748a8eb8',
    'config.json': 'c86d63c206a72695791bc202f42351c3fe825589d61aa6622a172f3a69642571',
    'source/mask2former/__init__.py': '6dafbf4e165fc117d6509d2bb54f628d367b1a3b7d602b82073743d63b45542d',
    'source/mask2former/mask2former_model.py': '9ca002f7f73ba4fe412dba0b4772f2e702633bc4d9bdc6e090dd149fab97944f',
    'source/mask2former/mask2former_pipeline.py': '9f78425bbe320df9f83e9f13af6e858cdee9da4a0a1aa63a6ed7bea861834bdb',
    'source/UPSTREAM_README.md': 'e1e997588a050e28ec6afd9ad17f4abd1ddfbba0979a9137b8666ffe9ca89e71',
}
PRECISION = {
    'sky': '2048-FP32-native-runtime-TF32', 'see': 'raw19-native-checkpoint-dtype-no-autocast-white+optional-original',
    'm2f': 'v4-768-FP32-TF32off', 'sam': '2.1-HieraLarge-CUDA-BF16-autocast',
    'preprocessing': 'sky-center-pad-white-round-see-native-dual-view-m2f-bilinear-imagenet-v2',
}


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _verified_hash(path, expected):
    actual = _digest(path)
    if actual != expected:
        raise ValueError('Model/source hash mismatch: ' + str(path))
    return actual


def _validate_rgb(rgb):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError('RGB must be a nonempty uint8 HxWx3 array')
    return rgb.shape[:2]


def _prepare_sky(rgb):
    h0, w0 = _validate_rgb(rgb)
    size = 2048
    h, w = (size, max(1, int(size*w0/h0))) if h0 > w0 else (max(1, int(size*h0/w0)), size)
    top, left = (size-h)//2, (size-w)//2
    padded = np.zeros((size, size, 3), np.float32)
    padded[top:top+h, left:left+w] = cv2.resize(rgb.astype(np.float32)/255, (w, h))
    return torch.from_numpy(padded.transpose(2, 0, 1)[None].copy()), (h, w, top, left)


def _preprocess_m2f(rgb):
    # Exact upstream PIL bilinear stretch, RGB /255 and ImageNet normalization.
    pixels = np.asarray(Image.fromarray(rgb).resize((768, 768), Image.Resampling.BILINEAR), dtype=np.float32)/255
    pixels = (pixels - np.array([.485, .456, .406], np.float32))/np.array([.229, .224, .225], np.float32)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()


@contextmanager
def _fp32_context(device):
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(torch.device(device).type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def _reject_training_loss(module, args):
    raise RuntimeError('Training loss must not be called during inference')


def _strict_load_m2f(model, state, *, expected_count=782):
    """Load every tensor, including the unused checkpoint loss-weight buffer."""
    normalized = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    if len(normalized) != len(state):
        raise RuntimeError('Strict checkpoint diagnostics: duplicate normalized keys')
    loss_keys = [k for k in normalized if k.endswith('criterion.empty_weight')]
    if len(loss_keys) != 1 or normalized[loss_keys[0]].shape not in [(13,), (151,)]:
        raise RuntimeError('Strict checkpoint diagnostics: invalid training loss buffer')
    loss = normalized[loss_keys[0]]
    model.criterion.register_buffer('empty_weight', torch.empty(loss.shape, dtype=loss.dtype))
    target = model.state_dict()
    mappings = {
        'identity': normalized,
        'strip_model': {k.removeprefix('model.'): v for k, v in normalized.items()},
        'add_model': {'model.'+k: v for k, v in normalized.items()},
    }
    diagnostics = {}
    for name, values in mappings.items():
        common = set(values) & set(target)
        diagnostics[name] = {
            'checkpoint_count': len(values), 'target_count': len(target), 'expected_count': expected_count,
            'missing': sorted(set(target)-set(values))[:12], 'extra': sorted(set(values)-set(target))[:12],
            'shape_mismatch': [(k, list(values[k].shape), list(target[k].shape)) for k in sorted(common) if values[k].shape != target[k].shape][:20],
        }
        if len(values) == expected_count and set(values) == set(target) and all(values[k].shape == target[k].shape for k in target):
            model.load_state_dict(values, strict=True, assign=True)
            if any(p.is_meta for p in model.parameters()) or any(b.is_meta for b in model.buffers()):
                raise RuntimeError('Strict checkpoint diagnostics: unresolved meta tensors')
            model.criterion.register_forward_pre_hook(_reject_training_loss)
            return {'key_mapping': name, 'tensor_count': len(values), 'strict_load': True,
                    'training_loss_buffer_shape': list(loss.shape), 'training_loss_forbidden_during_inference': True}
    raise RuntimeError('Strict checkpoint diagnostics: ' + json.dumps(diagnostics))


def _file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _import_path(path):
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        sys.path.remove(value)


def _box_pixels(box, h, w):
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > [w-1, h-1, w, h]) or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError('Box must be finite ordered pixel XYXY within the image')
    return values.astype(np.float32)


def _prompt_pixels(prompt, h, w):
    if not isinstance(prompt, dict):
        raise ValueError('Prompt must be a dict of pixel positive/negative points and/or box')
    arrays = []
    for name in ['positive', 'negative']:
        values = np.asarray(prompt.get(name, []), dtype=np.float64)
        if values.shape == (0,):
            values = values.reshape(0, 2)
        if values.ndim != 2 or values.shape[1:] != (2,) or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > [w-1, h-1]):
            raise ValueError('Points must be finite Nx2 pixel XY within the image')
        arrays.append(values.astype(np.float32))
    box = _box_pixels(prompt['box'], h, w) if prompt.get('box') is not None else None
    if not len(arrays[0]) and box is None:
        raise ValueError('A positive point or box is required')
    points = np.concatenate(arrays)
    labels = np.array([1]*len(arrays[0]) + [0]*len(arrays[1]), np.int32)
    return (points if len(points) else None), (labels if len(points) else None), box


class FusionModels:
    def __init__(self, assets=None, device='cuda', anime_assets=None, *, defer_checkpoint_checks=False):
        root = model_root()
        self.assets = Path(assets or root/'v2').resolve()
        self.anime_assets = Path(anime_assets or root/'AnimeSeg').resolve()
        self.device = str(torch.device(device))
        if torch.device(self.device).type not in ['cpu', 'cuda']:
            raise ValueError('FusionModels supports explicit cpu or cuda devices')
        self.sky = self.parts = self.m2f = self.sam = None
        self._sam_image_key = None
        self._m2f_info = {}
        self.identities = self._verify_assets(defer_checkpoint_checks=defer_checkpoint_checks)
        self._signature_base = {'assets': self.identities, 'precision': PRECISION, 'device': self.device,
                                'adapter_sha256': _digest(__file__)}

    @property
    def signature(self):
        # Sky retains the legacy runtime TF32 policy. Bind cache identity to that
        # actual policy even if the caller changes it between images.
        identity = {**self._signature_base, 'sky_tf32': self._tf32_flags()}
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _tf32_flags():
        return {'matmul': bool(torch.backends.cuda.matmul.allow_tf32), 'cudnn': bool(torch.backends.cudnn.allow_tf32)}

    def _verify_assets(self, *, defer_checkpoint_checks=False):
        def identity(path, expected):
            if defer_checkpoint_checks and path.suffix in ('.pt', '.safetensors'):
                # Builders still verify every checkpoint before loading it.
                if not path.is_file():
                    raise FileNotFoundError(path)
                if (not isinstance(expected, str) or len(expected) != 64
                        or any(char not in '0123456789abcdef' for char in expected)):
                    raise ValueError('Invalid checkpoint hash: ' + str(path))
                return expected
            return _verified_hash(path, expected)

        recorded = {k.replace('\\', '/'): v for k, v in json.loads((self.assets/'asset_hashes.json').read_text()).items()}
        result = {}
        for name in ['skytnt/isnet.py', 'skytnt/model.safetensors', 'see_through/checkpoint-18000.pt', 'sam2/sam2.1_hiera_large.pt']:
            result['v2/'+name] = identity(self.assets/name, recorded[name])
        for name, expected in PINNED_ANIME.items():
            result['anime/'+name] = identity(self.anime_assets/name, expected)
        for root, prefix in [(self.assets, 'v2/'), (self.anime_assets/'base-configs/large', 'anime/base-configs/large/')]:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in ['.py', '.yaml', '.yml', '.json'] and path.name != 'asset_hashes.json':
                    result[prefix+path.relative_to(root).as_posix()] = _digest(path)
        return result

    def _build_sky(self):
        from safetensors.torch import load_file
        _verified_hash(self.assets/'skytnt/model.safetensors', self.identities['v2/skytnt/model.safetensors'])
        module = _file_module('v21_isnet', self.assets/'skytnt/isnet.py')
        model = module.ISNetDIS().eval()
        state = load_file(str(self.assets/'skytnt/model.safetensors'), device='cpu')
        model.load_state_dict({k.removeprefix('net.'): v for k, v in state.items() if k.startswith('net.')}, strict=True)
        return model.to(device=self.device, dtype=torch.float32)

    def _build_see(self):
        from accelerate import init_empty_weights
        self._verify_see_imports()
        path = self.assets/'see_through/checkpoint-18000.pt'
        _verified_hash(path, self.identities['v2/see_through/checkpoint-18000.pt'])
        with _import_path(self.assets/'see_through/common'):
            from modules.semanticsam import SemanticSam
        self._verify_see_imports()
        state = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']
        with init_empty_weights(include_buffers=False):
            model = SemanticSam(class_num=19)
        model.load_state_dict(state, strict=True, assign=True)
        # Keep the existing checkpoint precision, including BF16, on all devices.
        return model.eval().to(self.device)

    def _verify_see_imports(self):
        """Reject Python's cached packages when they do not match signed See sources."""
        root = (self.assets/'see_through/common').resolve()
        for name, module in sorted(sys.modules.copy().items()):
            if name not in ['modules', 'utils'] and not name.startswith(('modules.', 'utils.')):
                continue
            expected = root.joinpath(*name.split('.'))
            filename = getattr(module, '__file__', None)
            if filename is None:
                # See's utils directory is an upstream namespace package.
                paths = {Path(p).resolve() for p in getattr(module, '__path__', [])}
                if paths != {expected} or not expected.is_dir():
                    raise RuntimeError('See source origin mismatch: ' + name)
                continue
            path = Path(filename).resolve()
            if path not in [expected.with_suffix('.py'), expected/'__init__.py']:
                raise RuntimeError('See source origin mismatch: ' + name)
            identity = self.identities.get('v2/'+path.relative_to(self.assets).as_posix())
            if identity is None:
                raise RuntimeError('See source is not signed: ' + name)
            _verified_hash(path, identity)

    def _build_m2f(self):
        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        for name, expected in PINNED_ANIME.items():
            _verified_hash(self.anime_assets/name, expected)
        source = self.anime_assets/'source/mask2former/mask2former_model.py'
        cls = _file_module('v21_mask2former_model', source).Mask2FormerAnimeSegModel
        with init_empty_weights(include_buffers=False):
            model = cls(str(self.anime_assets/'base-configs/large'), num_classes=12, load_base_pretrained=False)
        state = load_file(str(self.anime_assets/'models/anime_seg_mask2former_v4.safetensors'), device='cpu')
        self._m2f_info = _strict_load_m2f(model.model, state)
        return model.eval().to(device=self.device, dtype=torch.float32)

    def _build_sam(self):
        path = self.assets/'sam2/sam2.1_hiera_large.pt'
        _verified_hash(path, self.identities['v2/sam2/sam2.1_hiera_large.pt'])
        with _import_path(self.assets/'sam2/source'):
            import sam2
            if Path(sam2.__file__).resolve().parent != (self.assets/'sam2/source/sam2').resolve():
                raise RuntimeError('SAM2 source shadowed by another installed package')
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml', str(path), device=self.device, apply_postprocessing=False)
            return SAM2ImagePredictor(model, max_hole_area=0, max_sprinkle_area=0)

    def _build_generator(self):
        with _import_path(self.assets/'sam2/source'):
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        return SAM2AutomaticMaskGenerator(self.sam.model, points_per_side=8, points_per_batch=16,
                                         crop_n_layers=0, min_mask_region_area=0,
                                         max_hole_area=0, max_sprinkle_area=0)

    def _autocast(self, dtype=torch.bfloat16):
        return torch.autocast(torch.device(self.device).type, dtype=dtype) if dtype in [torch.float16, torch.bfloat16] else nullcontext()

    def _clear(self, name):
        value = getattr(self, name)
        if name == 'sam':
            if value is not None:
                value.reset_predictor()
            self._sam_image_key = None
        if value is None:
            return
        setattr(self, name, None)
        del value
        gc.collect()
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()

    def release(self):
        had_model = any(value is not None for value in (self.sky, self.parts, self.m2f, self.sam))
        for name in ['sky', 'parts', 'm2f', 'sam']:
            self._clear(name)
        if not had_model and self.device.startswith('cuda'):
            # The denoiser can also leave free blocks in the CUDA allocator.
            torch.cuda.empty_cache()

    def segment_evidence(self, rgb_uint8, *, dual_view=True):
        h0, w0 = _validate_rgb(rgb_uint8)
        if type(dual_view) is not bool:
            raise ValueError('dual_view must be a bool')
        self.release()
        provenance = {'input_shape': list(rgb_uint8.shape), 'grid': 'EXIF-oriented-original',
                      'model_hashes': self.identities, 'device': self.device, 'signature': self.signature}
        tensor, (h, w, top, left) = _prepare_sky(rgb_uint8)
        try:
            self.sky = self._build_sky()
            with torch.inference_mode(), torch.autocast(torch.device(self.device).type, enabled=False):
                prediction = self.sky(tensor.to(self.device))[0][0].sigmoid()
                values = prediction.float().cpu().numpy()[0, 0]
            alpha = cv2.resize(values[top:top+h, left:left+w], (w0, h0)).clip(0, 1).astype(np.float32)
            if not np.isfinite(alpha).all():
                raise RuntimeError('SkyTNT returned nonfinite alpha')
            provenance['sky'] = {'precision': 'float32', 'input_shape': [1, 3, 2048, 2048],
                                 'resized_shape': [h, w], 'padding_top_left': [top, left],
                                 'tf32': self._tf32_flags(),
                                 'preprocessing': 'RGB /255, bilinear preserve aspect, centered zero padding'}
            del prediction, values
        finally:
            del tensor
            self._clear('sky')
        white = np.rint(rgb_uint8.astype(np.float32)*alpha[..., None]+255*(1-alpha[..., None])).clip(0, 255).astype(np.uint8)
        see_results = {}
        see_preprocessing = {'white': 'Sky white rounded RGB; native See resize/pad/normalization'}
        if dual_view:
            see_preprocessing['original'] = 'Original RGB without Sky foreground removal; native See resize/pad/normalization'
        try:
            self.parts = self._build_see()
            dtype = next(self.parts.parameters()).dtype
            with torch.inference_mode(), torch.autocast(torch.device(self.device).type, enabled=False):
                for name in see_preprocessing:
                    # Keep the caller's original pixels independent of model
                    # preprocessing while retaining both unmerged score views.
                    pixels = white if name == 'white' else rgb_uint8.copy()
                    values = self.parts.inference(pixels)[0].float().cpu().numpy()
                    if values.shape != (19, h0, w0) or not np.isfinite(values).all():
                        raise RuntimeError('See ' + name + ' must return finite raw19 logits on the original grid')
                    see_results[name] = values
                    del pixels, values
            provenance['see'] = {'checkpoint_dtype': str(dtype).removeprefix('torch.'), 'output_dtype': 'float32',
                                 'input_shape': list(white.shape), 'encoder_size': int(self.parts.img_size),
                                 'preprocessing': see_preprocessing['white'],
                                 'views': list(see_preprocessing), 'view_preprocessing': see_preprocessing,
                                 'autocast': 'disabled'}
        finally:
            self._clear('parts')
        results = {}
        try:
            self.m2f = self._build_m2f()
            for name, pixels in [('m2f_white', white)] + ([('m2f_original', rgb_uint8)] if dual_view else []):
                tensor = _preprocess_m2f(pixels).to(self.device)
                with torch.inference_mode(), _fp32_context(self.device):
                    output = self.m2f(tensor)
                    values = output['semantic_logits']
                    if values.shape != (1, 12, 768, 768) or not torch.isfinite(values).all():
                        raise RuntimeError('M2F must return finite 12x768x768 semantic log probabilities')
                    results[name] = values.float().log_softmax(1)[0].cpu().numpy()
                del tensor, output, values
            provenance['m2f'] = {**self._m2f_info, 'version': 4, 'input_shape': [1, 3, 768, 768],
                                 'precision': 'float32', 'tf32': False, 'views': ['white', 'original'] if dual_view else ['white'],
                                 'preprocessing': 'upstream PIL bilinear stretch RGB /255 ImageNet mean/std',
                                 'postprocessing': 'normalized log probabilities; native argmax then PIL nearest for reference'}
        finally:
            self._clear('m2f')
        native_labels = results['m2f_white'].argmax(0).astype(np.uint8)
        reference = np.asarray(Image.fromarray(native_labels).resize((w0, h0), Image.Resampling.NEAREST)).copy()
        return {'alpha': alpha, 'see_logits': see_results['white'],
                'see_original_logits': see_results.get('original'),
                'm2f_white': results['m2f_white'], 'm2f_original': results.get('m2f_original'),
                'reference_labels12': reference, 'provenance': provenance}

    def _ensure_sam(self):
        for name in ['sky', 'parts', 'm2f']:
            self._clear(name)
        if self.sam is None:
            self.sam = self._build_sam()

    def predict_candidates(self, rgb_uint8, prompt, *, mask_input=None):
        h, w = _validate_rgb(rgb_uint8)
        points, labels, box = _prompt_pixels(prompt, h, w)
        if mask_input is not None and (not isinstance(mask_input, np.ndarray) or mask_input.dtype != np.float32 or mask_input.shape != (1, 256, 256) or not np.isfinite(mask_input).all()):
            raise ValueError('mask_input must be finite float32[1,256,256] logits')
        self._ensure_sam()
        key = (rgb_uint8.shape, hashlib.sha256(rgb_uint8.tobytes()).hexdigest())
        context = self._autocast() if self.device.startswith('cuda') else nullcontext()
        with torch.inference_mode(), context:
            if key != self._sam_image_key:
                self.sam.set_image(rgb_uint8.copy())
                self._sam_image_key = key
            values, scores, lowres = self.sam.predict(point_coords=points, point_labels=labels, box=box,
                                                     mask_input=mask_input, multimask_output=True, return_logits=True)
        values, scores, lowres = [np.asarray(value, dtype=np.float32) for value in [values, scores, lowres]]
        if values.shape != (3, h, w) or scores.shape != (3,) or lowres.shape != (3, 256, 256) or not all(np.isfinite(a).all() for a in [values, scores, lowres]):
            raise RuntimeError('SAM returned invalid candidate shapes or nonfinite values')
        threshold = float(self.sam.mask_threshold)
        intersection = (values > threshold+1).sum(axis=(1, 2), dtype=np.int64)
        union = (values > threshold-1).sum(axis=(1, 2), dtype=np.int64)
        stability = np.divide(intersection, union, out=np.zeros(3, np.float32), where=union > 0)
        return {'masks': values > threshold, 'scores': scores, 'lowres': lowres, 'stability': stability.astype(np.float32)}

    def generate_candidates(self, rgb_uint8, box):
        h, w = _validate_rgb(rgb_uint8)
        x0, y0, x1, y1 = _box_pixels(box, h, w)
        x0, y0, x1, y1 = int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))
        self._ensure_sam()
        generator = self._build_generator()
        context = self._autocast() if self.device.startswith('cuda') else nullcontext()
        try:
            with torch.inference_mode(), context:
                proposals = generator.generate(rgb_uint8[y0:y1, x0:x1].copy())
        finally:
            # AMG owns a separate image predictor; never retain its embedding.
            if hasattr(generator, 'predictor'):
                generator.predictor.reset_predictor()
            del generator
        results = []
        for proposal in proposals:
            local = np.asarray(proposal['segmentation'])
            if local.dtype != np.bool_ or local.shape != (y1-y0, x1-x0):
                raise RuntimeError('SAM ROI generator returned an invalid mask')
            score, stability = float(proposal['predicted_iou']), float(proposal['stability_score'])
            if not np.isfinite([score, stability]).all():
                raise RuntimeError('SAM ROI generator returned nonfinite scores')
            yy, xx = np.nonzero(local)
            if not len(xx):
                continue
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = local
            results.append({'mask': mask, 'score': score, 'stability': stability,
                            'bbox': [int(xx.min())+x0, int(yy.min())+y0, int(xx.max())+x0+1, int(yy.max())+y0+1]})
        return results
