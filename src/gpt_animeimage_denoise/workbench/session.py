"""Keep one analysed image and its denoising candidates available to the workbench."""
from collections import OrderedDict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from ..denoisers import (DOWNSCALE, describe_denoiser, resolve_denoiser,
                         smooth_candidate, super_resolve, uses_l0, uses_sr)
from ..editor import SelectionEditor
from ..fusion_evaluation import _review_candidate
from ..pipeline import _assert_inputs_safe, _case_folder, _digest, _read_source, _sha
from ..preview import _verified_output


def _completed_case_source(source, case, report, digest):
    if report['status'] != 'complete' or report['profile'] != 'fusion' or report['edits']:
        raise ValueError('Workbench requires a completed fusion case without manual corrections')
    if report['source_sha256'] != digest:
        raise ValueError('Source changed while preparing the workbench')
    if 'original.png' in report['output_hashes']:
        original = _verified_output(case, report, 'original.png')
    else:
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('本地原图副本已更改，请重新打开源图。')
        original = BytesIO(content)
    rgb, alpha, icc = _read_source(original)
    if report['size'] != [rgb.shape[1], rgb.shape[0]]:
        raise ValueError('Workbench image dimensions differ from the completed report')
    return rgb, alpha, icc


def _cached_l0_baseline(source, workspace, digest, size, lam):
    """Read an existing matching float candidate without opening the models."""
    output = _case_folder(Path(workspace).resolve(), digest)
    case = _case_folder(output, 'analysis')
    report = json.loads((case / 'report.json').read_text(encoding='utf-8'))
    parameters = report['parameters']
    device = parameters['device']
    if (report['folder'] != 'analysis' or report['size'] != size
            or parameters.get('denoiser', 'l0') != 'l0' or parameters['lambda'] != lam
            or not isinstance(device, str) or (device != 'cpu' and not device.startswith('cuda'))):
        raise ValueError('Workbench L0 cache does not match the requested source and settings')
    rgb, alpha, icc = _completed_case_source(source, case, report, digest)
    return rgb, alpha, icc, _review_candidate(output / '.cache', report, rgb.shape)


def _case_editor(source, case, report, digest):
    rgb, alpha, icc = _completed_case_source(source, case, report, digest)
    with Image.open(_verified_output(case, report, 'person_labels.png')) as image:
        labels = np.array(image)
    with Image.open(_verified_output(case, report, 'objects.png')) as image:
        objects = np.array(image.convert('L')) > 0
    with Image.open(_verified_output(case, report, 'unknown.png')) as image:
        unknown = np.array(image.convert('L')) > 0
    return SelectionEditor(rgb, alpha, labels, objects, unknown), icc


def _cached_editor(source, workspace, digest, size):
    """Read a verified analysis grid without loading a denoising candidate."""
    case = _case_folder(_case_folder(Path(workspace).resolve(), digest), 'analysis')
    report = json.loads((case / 'report.json').read_text(encoding='utf-8'))
    if report['folder'] != 'analysis' or report['size'] != size:
        raise ValueError('Workbench analysis does not match the requested source grid')
    return _case_editor(source, case, report, digest)[0]


class WorkbenchSession:
    """Use immutable analysis inputs while region choices and denoisers change."""

    def __init__(self, source, case_dir, source_sha256, device, editor, icc, baseline, baseline_lam,
                 *, models, suspected_objects=None, baseline_method='l0', baseline_downscale=DOWNSCALE):
        self.source = source
        self.case_dir = case_dir
        self.name = source.name
        self.source_sha256 = source_sha256
        self.device = device
        self.editor = editor
        self.suspected_objects = suspected_objects
        self._models = models
        self.icc = icc
        self.height, self.width = editor.labels.shape
        if baseline is not None:
            baseline.setflags(write=False)
        self._baseline = baseline
        self._baseline_key = self._candidate_key(baseline_method, baseline_lam, baseline_downscale)
        # Keep the original candidate, two recent alternatives, and one SR input.
        self._candidates = OrderedDict()
        self._sr_input = None
        self._sr_key = None

    @classmethod
    def open(cls, source, workspace, *, device='auto', method='l0', lam=None, downscale=DOWNSCALE, on_stage=None):
        """Run or resume a fusion case using the document's actual denoiser."""
        import torch

        from ..fusion_models import FusionModels
        from ..fusion_pipeline import process_image

        source = Path(source).resolve(strict=True)
        digest = _sha(source)
        output = _case_folder(Path(workspace).resolve(), digest)
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if device != 'cpu' and not device.startswith('cuda'):
            raise ValueError('device must be auto/cpu/cuda/cuda:N')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu')

        models = FusionModels(device=device, defer_checkpoint_checks=True)
        try:
            report = process_image(source, output, models, case_id='analysis', profile='fusion',
                                   denoiser=method, lam=lam, downscale=downscale, on_stage=on_stage, _workbench=True)
        finally:
            models.release()
        if _sha(source) != digest:
            raise ValueError('Source changed while preparing the workbench')
        return cls._load_completed_case(source, output, report, digest, downscale, models=models)

    @classmethod
    def from_cache(cls, source, workspace, digest, size, *, method='l0', lam=None, downscale=DOWNSCALE):
        """Restore the analysis grid, and reuse the float candidate when settings match."""
        output = _case_folder(Path(workspace).resolve(), digest)
        case = _case_folder(output, 'analysis')
        report = json.loads((case / 'report.json').read_text(encoding='utf-8'))
        parameters = report['parameters']
        device = parameters['device']
        method, lam = resolve_denoiser(method, lam)
        if (report['folder'] != 'analysis' or report['size'] != size
                or not isinstance(device, str) or not (device == 'cpu' or device.startswith('cuda'))):
            raise ValueError('Workbench cache does not match the requested source grid')
        matching = (parameters.get('denoiser', 'l0') == method
                    and (not uses_l0(method) or parameters['lambda'] == lam)
                    and parameters.get('denoiser_config') == describe_denoiser(method, device, downscale=downscale))
        return cls._load_completed_case(Path(source), output, report, digest, downscale, models=None,
                                        load_candidate=matching)

    @classmethod
    def _load_completed_case(cls, source, output, report, digest, downscale, *, models, load_candidate=True):
        case = _case_folder(output, report['folder'])
        editor, icc = _case_editor(source, case, report, digest)
        suspected = None
        if 'suspected_objects.png' in report['output_hashes']:
            with Image.open(_verified_output(case, report, 'suspected_objects.png')) as image:
                suspected = np.array(image.convert('L')) > 0
            if suspected.shape != editor.labels.shape or (suspected & editor.objects).any():
                raise ValueError('Suspected object mask must align with the image and exclude confirmed objects')
            suspected.setflags(write=False)
        baseline = None
        if load_candidate:
            try:
                baseline = _review_candidate(output / '.cache', report, editor.rgb.shape)
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                if models is not None:
                    raise
                # A damaged float cache does not invalidate the verified masks.
        return cls(source, case, digest, report['parameters']['device'], editor, icc, baseline,
                   report['parameters']['lambda'], models=models, suspected_objects=suspected,
                   baseline_method=report['parameters'].get('denoiser', 'l0'), baseline_downscale=downscale)

    def _inference_device(self):
        if self._models is None:
            import torch

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return self.device

    def predict_object(self, positive, negative):
        """Preview a manually prompted shape in the source pixel grid."""
        from ..auto_sam import select_manual_candidate

        rgb, alpha = self.editor.rgb, self.editor.alpha
        if alpha is not None:
            if any(alpha[y, x] == 0 for x, y in positive):
                raise ValueError('目标点不能落在完全透明的位置。')
            opacity = alpha.astype(np.float32)[..., None] / 255
            rgb = np.rint(rgb * opacity + 255 * (1 - opacity)).astype(np.uint8)
        prompt = {'positive': positive, 'negative': negative}
        if self._models is None:
            from ..fusion_models import FusionModels

            self._models = FusionModels(device=self._inference_device(), defer_checkpoint_checks=True)
        try:
            prediction = self._models.predict_candidates(rgb, prompt)
            mask, _ = select_manual_candidate(prediction, prompt)
        finally:
            self._models.release()
        if mask is None:
            raise ValueError('SAM2 没有找到符合这些点的轮廓，请调整加入点或排除点。')
        return mask if alpha is None else mask & (alpha > 0)

    @staticmethod
    def _candidate_key(method, lam, downscale):
        return method, lam if uses_l0(method) else None, downscale if method.startswith('esrgan') else None

    def candidate(self, method='l0', *, lam=None, downscale=DOWNSCALE):
        """Return the original-grid float32 candidate for the selected method."""
        method, lam = resolve_denoiser(method, lam)
        if self._baseline is not None and self._candidate_key(method, lam, downscale) == self._baseline_key:
            return self._baseline

        device = self._inference_device()
        sr_key = None
        if uses_sr(method):
            config = describe_denoiser(method, device, downscale=downscale)
            sr_key = _digest({name: value for name, value in config.items() if name != 'method'})
        l0_enabled = uses_l0(method)
        key = (method, lam if l0_enabled else None, sr_key)
        if key in self._candidates:
            self._candidates.move_to_end(key)
            return self._candidates[key]

        rgb = self.editor.rgb
        if sr_key is not None:
            if sr_key != self._sr_key:
                self._sr_input = super_resolve(rgb, device=device, method=method, downscale=downscale)
                self._sr_key = sr_key
            rgb = self._sr_input
        candidate = (smooth_candidate(rgb, lam, device=device) if l0_enabled
                     else rgb.astype(np.float32) / np.float32(255))
        candidate.setflags(write=False)
        self._candidates[key] = candidate
        if len(self._candidates) > 2:
            self._candidates.popitem(last=False)
        return candidate

    def save_corrections(self, directory, modules):
        """Export portable, source-bound edits in a new or empty directory."""
        directory = Path(directory)
        if directory.is_symlink():
            raise ValueError('Correction output must not be a symbolic link')
        directory = directory.resolve()
        allow_path = directory / 'allow.png'
        protect_path = directory / 'protect.png'
        json_path = directory / 'corrections.json'
        _assert_inputs_safe([allow_path, protect_path, json_path], [self.source])
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise FileExistsError('Choose a new or empty correction directory')
        if _sha(self.source) != self.source_sha256:
            raise ValueError('Source changed after analysis; reopen it before saving corrections')
        allow, protect = self.editor.correction_masks(modules)
        record = {'version': 2, 'source_sha256': self.source_sha256,
                  'size': [self.width, self.height], 'allow_mask': allow_path.name,
                  'protect_mask': protect_path.name}
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        try:
            for mask, path in ((allow, allow_path), (protect, protect_path)):
                with path.open('xb') as stream:
                    written.append(path)
                    Image.fromarray(mask.astype(np.uint8) * 255).save(stream, format='PNG')
            with json_path.open('x', encoding='utf-8') as stream:
                written.append(json_path)
                json.dump(record, stream, indent=2)
                stream.write('\n')
        except BaseException:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        return json_path
