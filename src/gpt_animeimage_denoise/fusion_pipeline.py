"""V2.1 original-grid orchestration, independent float caches and JSON/CSV output."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import time
from zipfile import BadZipFile

import numpy as np
from PIL import Image

from .auto_sam import refine_regions, select_manual_candidate
from .corrections import MODULES, apply_corrections, load_edits
from .filters import blend, build_weight, gradient_preserving_candidate, validate_unit_interval
from .denoisers import (describe_denoiser, resolve_denoiser, uses_sr, uses_l0,
                        super_resolve, smooth_candidate)
from .fusion import _policy, fuse_evidence, reference_regions
from .pipeline import (_assert_inputs_safe, _case_folder, _digest,
                       _json, _l0_callable_signature, _read_source, _save_rgb, _sha)

OUTPUT_FILES = ['original.png', 'result.png', 'person_labels.png', 'raw_labels.png',
                'objects.png', 'suspected_objects.png', 'unknown.png', 'conflict.png', 'consensus.png',
                'protected.png', 'source_map.png', 'weight.npy', 'weight.png',
                'preview.png', 'regions.json', *['module_' + name + '.png' for name in MODULES]]

DEFAULT_STRUCTURE_PROTECTION = 0.2
DEFAULT_RESTORATION_GAIN = 0.25


def _code_hash(names):
    root = Path(__file__).resolve().parent
    return _digest({name: _sha(root / name) for name in names})


def _targets(output, folder):
    return [folder / name for name in [*OUTPUT_FILES, 'report.json', 'report.json.tmp', 'regions.json.tmp']] + [
        output / name for name in ('records.json', 'records.json.tmp', 'summary.csv', 'summary.csv.tmp')]


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        key = 'array_' + str(len(arrays)); arrays[key] = value
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError('Cache arrays must be finite and non-object')
        return {'__ndarray__': key}
    if isinstance(value, dict):
        return {key: _pack(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__ndarray__'}:
            return arrays[value['__ndarray__']].copy()
        return {key: _unpack(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item, arrays) for item in value]
    return value


def _payload_hash(tree, arrays):
    return _digest({'tree': tree, 'arrays': {key: {'shape': list(value.shape),
        'dtype': value.dtype.str, 'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
        for key, value in arrays.items()}})


def _cached(output, stage, identity, compute, validate, warnings, inputs):
    cache = output / '.cache'
    path = cache / (stage + '-' + _digest(identity) + '.npz')
    temporary = path.with_suffix('.tmp')
    _assert_inputs_safe([path, temporary], inputs)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata'].item()))
                if metadata['identity'] != identity:
                    raise ValueError('cache identity mismatch')
                arrays = {key: saved[key] for key in saved.files if key != 'metadata'}
                if any(value.dtype.hasobject or not np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError('nonfinite or object cache array')
                if metadata['payload_hash'] != _payload_hash(metadata['tree'], arrays):
                    raise ValueError('cache payload hash mismatch')
                value = _unpack(metadata['tree'], arrays)
            validate(value)
            return value, True
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as error:
            warnings.append(stage + ' cache rejected and recomputed: ' + type(error).__name__)
    if compute is None:
        return None, False
    value = compute(); validate(value)
    arrays = {}; tree = _pack(value, arrays)
    metadata = {'identity': identity, 'tree': tree, 'payload_hash': _payload_hash(tree, arrays)}
    cache.mkdir(parents=True, exist_ok=True)
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata, sort_keys=True, allow_nan=False), **arrays)
    temporary.replace(path)
    return value, False


def _validate_evidence(evidence, shape, dual_view, *, require_original_see=None):
    required = {'alpha': shape, 'see_logits': (19, *shape), 'm2f_white': (12, 768, 768),
                'reference_labels12': shape}
    if dual_view:
        required['m2f_original'] = (12, 768, 768)
    see_views = evidence.get('provenance', {}).get('see', {}).get('views', [])
    if require_original_see is None:
        require_original_see = dual_view
    if require_original_see or evidence.get('see_original_logits') is not None or 'original' in see_views:
        required['see_original_logits'] = (19, *shape)
    for name, expected in required.items():
        value = evidence.get(name)
        if not isinstance(value, np.ndarray) or value.shape != expected or not np.isfinite(value).all():
            raise ValueError('Invalid evidence shape/values: ' + name)
    for name in ('alpha', 'see_logits', 'see_original_logits', 'm2f_white', 'm2f_original'):
        if evidence.get(name) is not None and evidence[name].dtype != np.float32:
            raise ValueError('Evidence must retain float32: ' + name)
    if evidence['reference_labels12'].dtype != np.uint8 or np.any(evidence['reference_labels12'] > 11):
        raise ValueError('Invalid canonical reference labels')
    if not isinstance(evidence.get('provenance'), dict):
        raise ValueError('Evidence provenance missing')


def _validate_candidates(value, shape):
    for name, expected in [('masks', (3, *shape)), ('scores', (3,)), ('stability', (3,)), ('lowres', (3, 256, 256))]:
        array = value[name]
        if not isinstance(array, np.ndarray) or array.shape != expected or not np.isfinite(array).all():
            raise ValueError('Invalid SAM candidates: ' + name)
    if value['masks'].dtype != bool:
        raise ValueError('SAM masks must be bool')
    for name in ('scores', 'stability', 'lowres'):
        if value[name].dtype != np.float32:
            raise ValueError('SAM candidate arrays must retain float32: ' + name)


class _CandidateCache:
    """Intercept SAM inference, retaining each float low-resolution candidate."""
    def __init__(self, models, output, identity, warnings, inputs):
        self.models = models; self.output = output; self.identity = identity
        self.warnings = warnings; self.inputs = inputs; self.hits = 0; self.misses = 0

    def __getattr__(self, name):
        return getattr(self.models, name)

    def predict_candidates(self, rgb, prompt, *, mask_input=None):
        arrays = {}; packed = _pack(mask_input, arrays)
        identity = dict(self.identity, prompt=prompt, mask_input=_payload_hash(packed, arrays))
        value, hit = _cached(self.output, 'sam', identity,
            lambda: self.models.predict_candidates(rgb, prompt, mask_input=mask_input),
            lambda value: _validate_candidates(value, rgb.shape[:2]), self.warnings, self.inputs)
        self.hits += int(hit); self.misses += int(not hit)
        return value

    def generate_candidates(self, rgb, box):
        identity = dict(self.identity, box=box, provider='roi')

        def validate(proposals):
            if not isinstance(proposals, list):
                raise ValueError('SAM ROI proposals must be a list')
            for proposal in proposals:
                mask = proposal.get('mask', proposal.get('masks'))
                if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != rgb.shape[:2]:
                    raise ValueError('SAM ROI proposals must be bool on original grid')
                if not math.isfinite(proposal['score']) or not math.isfinite(proposal['stability']):
                    raise ValueError('SAM ROI proposal quality must be finite')

        value, hit = _cached(self.output, 'sam-roi', identity,
            lambda: self.models.generate_candidates(rgb, box), validate, self.warnings, self.inputs)
        self.hits += int(hit); self.misses += int(not hit)
        return value


def _validate_regions(regions, shape):
    for name in ('person_labels', 'raw_labels', 'source_map'):
        value = regions[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.uint8:
            raise ValueError('Invalid region labels: ' + name)
    if not np.isin(regions['person_labels'], [0, 1, 2, 3, 4, 5, 6, 255]).all():
        raise ValueError('Person labels must be 0..6 or 255')
    for name in ('allowed', 'protected', 'objects', 'uncertain', 'conflict', 'consensus'):
        value = regions[name]
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != bool:
            raise ValueError('Invalid region mask: ' + name)
    if 'suspected_objects' in regions:
        value = regions['suspected_objects']
        if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != bool:
            raise ValueError('Invalid region mask: suspected_objects')


def _manual_sam(rgb, edits, models):
    masks = []; records = []; height, width = rgb.shape[:2]
    for entry in edits['sam_regions']:
        prompt = {}
        for name in ('positive', 'negative'):
            prompt[name] = [[min(float(x) * width, width - 1), min(float(y) * height, height - 1)]
                            for x, y in entry.get(name, [])]
        if 'box' in entry:
            x1, y1, x2, y2 = entry['box']
            prompt['box'] = [x1 * width, y1 * height, x2 * width, y2 * height]
        prediction = models.predict_candidates(rgb, prompt)
        mask, record = select_manual_candidate(prediction, prompt, candidate=entry.get('candidate'))
        records.append(dict(record, manual=True, action=entry['action'], part=entry.get('part')))
        if mask is not None:
            masks.append((entry, mask))
    return masks, records


def _resume(folder, fingerprint, warnings):
    path = folder / 'report.json'
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('status') != 'complete' or report.get('fingerprint') != fingerprint:
            return None
        if set(report['output_hashes']) != set(OUTPUT_FILES):
            raise ValueError('result inventory mismatch')
        if any(not (folder / name).is_file() or _sha(folder / name) != digest
               for name, digest in report['output_hashes'].items()):
            raise ValueError('result hash mismatch')
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.append('result cache rejected and recomputed: ' + type(error).__name__)
        return None
    return dict(report, resumed=True)


def _load_policy(policy, policy_path):
    if policy is not None and policy_path is not None:
        raise ValueError('policy and policy_path are mutually exclusive')
    path = None; digest = None
    if policy_path is not None:
        path = Path(policy_path).resolve()
        payload = path.read_bytes()
        policy = json.loads(payload)
        digest = hashlib.sha256(payload).hexdigest()
        if not isinstance(policy, dict):
            raise ValueError('Policy JSON must be an object')
    elif policy is not None and not isinstance(policy, dict):
        raise ValueError('policy must be a dictionary')
    actual = json.loads(json.dumps(_policy(policy), allow_nan=False))
    return actual, path, digest


def _finish_settings(profile, structure_protection, restoration_gain):
    if structure_protection is None:
        structure_protection = DEFAULT_STRUCTURE_PROTECTION if profile == 'fusion' else 1.0
    if restoration_gain is None:
        restoration_gain = DEFAULT_RESTORATION_GAIN if profile == 'fusion' else 1.0
    return (validate_unit_interval(structure_protection, 'structure_protection'),
            validate_unit_interval(restoration_gain, 'restoration_gain'))


def process_image(source, output_root, models, *, profile='fusion', case_id=None,
                  lam=None, sigma=12, strength=1, edits_path=None, split='',
                  policy=None, policy_path=None, l0_fn=None, denoiser='l0',
                  sr_fn=None, downscale=.6, structure_protection=None,
                  restoration_gain=None, _input_paths=(), on_stage=None):
    """Preserve source bytes, alpha, ICC and every final W=0 pixel exactly."""
    def stage(name):
        if on_stage is not None:
            on_stage(name)

    if profile not in ('fusion', 'reference'):
        raise ValueError('fusion_pipeline supports reference/fusion profiles only')
    denoiser, lam = resolve_denoiser(denoiser, lam)
    if not math.isfinite(lam) or lam <= 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('lambda and sigma must be finite positive values')
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite in 0..1')
    structure_protection, restoration_gain = _finish_settings(profile, structure_protection, restoration_gain)
    policy, policy_path, policy_hash = _load_policy(policy, policy_path)
    if profile == 'reference':
        policy = dict(policy, semantic_repair='off', outfit_reference='off')
    source = Path(source).resolve(); output = Path(output_root).resolve()
    input_hash = _sha(source)
    if case_id is None:
        stem = re.sub(r'[^\w-]+', '-', source.stem).strip('-')[:64] or 'image'
        case_id = stem + '-' + input_hash[:10]
    folder = _case_folder(output, case_id)
    rgb, original_alpha, icc = _read_source(source)
    edits = load_edits(edits_path, rgb.shape[:2], input_hash)
    policy_inputs = [policy_path] if policy_path is not None else []
    inputs = list(dict.fromkeys([source, *edits['input_paths'], *policy_inputs, *map(Path, _input_paths)]))
    _assert_inputs_safe(_targets(output, folder), inputs)
    input_snapshots = {str(path): _sha(path) for path in [source, *edits['input_paths']]}
    if policy_path is not None:
        input_snapshots[str(policy_path)] = policy_hash
    code = _code_hash(['fusion_pipeline.py', 'corrections.py', 'fusion.py', 'auto_sam.py',
                       'fusion_models.py', 'grouping.py', 'filters.py', 'pipeline.py', 'denoisers.py', 'semantic_repair.py',
                       '../denoise.py', '../l0_cuda.py', '../tiled_inference.py'])
    device = str(getattr(models, 'device', 'cpu'))
    stage('candidate')
    denoiser_config = describe_denoiser(denoiser, device, sr_fn=sr_fn, downscale=downscale)
    l0_implementation = (json.loads(json.dumps(_l0_callable_signature(l0_fn), allow_nan=False))
                         if uses_l0(denoiser) else None)
    sr_model_path = None
    if uses_sr(denoiser) and sr_fn is None:
        sr_model_path = Path(denoiser_config['model']['path']).resolve()
        inputs.append(sr_model_path)
        _assert_inputs_safe(_targets(output, folder), inputs)
        if sr_model_path.is_file():
            model_digest = _sha(sr_model_path)
            if model_digest != denoiser_config['model']['expected_sha256']:
                raise ValueError('Denoiser model checksum differs from configured model')
            input_snapshots[str(sr_model_path)] = model_digest
    stage('analysis')
    parameters = {'profile': profile, 'lambda': float(lam), 'sigma_at_2048': float(sigma),
                  'strength': float(strength), 'device': device, 'models': str(models.signature),
                  'structure_protection': structure_protection, 'restoration_gain': restoration_gain,
                  'implementation': code, 'policy': policy, 'policy_sha256': policy_hash,
                  'l0_implementation': l0_implementation,
                  'l0_enabled': uses_l0(denoiser),
                  'denoiser': denoiser, 'denoiser_config': denoiser_config}
    fingerprint = _digest({'source_sha256': input_hash, 'parameters': parameters, 'edits': edits['hashes']})
    warnings = []
    resumed = _resume(folder, fingerprint, warnings)
    if resumed is not None:
        if any(_sha(path) != digest for path, digest in input_snapshots.items()):
            raise RuntimeError('Source, edits or policy changed while processing; cached result not reused')
        resumed['split'] = split
        return resumed
    started = time.perf_counter()
    segment_rgb = rgb
    if original_alpha is not None:
        alpha = original_alpha.astype(np.float32)[..., None] / 255
        segment_rgb = np.rint(rgb * alpha + 255 * (1 - alpha)).astype(np.uint8)
    evidence_identity = {'source_sha256': input_hash, 'models': str(models.signature), 'device': device,
        'preprocessing': 'exif-original-grid-white-alpha-v1', 'dual_view': profile == 'fusion',
        'see_views': ['white', 'original'] if profile == 'fusion' else ['white'],
        'code': _code_hash(['fusion_models.py', 'fusion_pipeline.py', 'pipeline.py'])}
    evidence = None; evidence_hit = False
    if profile == 'reference':
        # Canonical white labels are identical in single/full evidence. Reuse
        # an existing validated full cache, without requesting a new dual view.
        full_identity = dict(evidence_identity, dual_view=True, see_views=['white', 'original'])
        evidence, evidence_hit = _cached(output, 'evidence', full_identity, None,
            lambda value: _validate_evidence(value, rgb.shape[:2], True), warnings, inputs)
        if evidence_hit:
            evidence_identity = full_identity
    if evidence is None:
        evidence, evidence_hit = _cached(output, 'evidence', evidence_identity,
            lambda: models.segment_evidence(segment_rgb, dual_view=profile == 'fusion'),
            lambda value: _validate_evidence(value, rgb.shape[:2], profile == 'fusion'), warnings, inputs)
    candidate_models = _CandidateCache(models, output, dict(evidence_identity, policy=policy,
        edits=edits['hashes'], sam_code=_code_hash(['auto_sam.py', 'corrections.py'])), warnings, inputs)
    if profile == 'reference':
        regions = reference_regions(evidence)
        regions['policy'] = dict(policy, profile='reference')
    else:
        regions = fuse_evidence(evidence, policy=policy)
        regions = refine_regions(segment_rgb, evidence, regions, candidate_models, policy=policy)
    _validate_regions(regions, rgb.shape[:2])
    manual_masks, manual_records = _manual_sam(segment_rgb, edits, candidate_models)
    regions = apply_corrections(regions, edits, manual_masks)
    regions['sam_records'] = [*regions.get('sam_records', []), *manual_records]
    unknown = regions['person_labels'] == 255
    suspected_objects = regions.get('suspected_objects', np.zeros_like(unknown))
    suspected_objects &= ~regions['objects']
    # The default scope stays skin/face/hair, even if an upstream allowed mask is broader.
    default_scope = np.isin(regions['person_labels'], [1, 2, 3])
    reviewed_allow = edits['allow_mask'].copy()
    for entry, mask in manual_masks:
        if entry['action'] == 'allow':
            reviewed_allow |= mask
    regions['allowed'] &= default_scope | reviewed_allow
    regions['protected'] |= unknown | regions['objects']
    if original_alpha is not None:
        regions['protected'] |= original_alpha == 0
    regions['allowed'] &= ~regions['protected']
    release = getattr(models, 'release', None)
    if release is not None:
        release()
    scale = min(rgb.shape[:2]) / 2048
    effective_sigma = float(sigma) * scale
    weight = build_weight(regions['allowed'], regions['protected'], rgb,
        feather_radius=16 * scale, structure_radius=5 * scale, gradient_threshold=4,
        structure_protection=structure_protection)
    weight *= np.float32(strength); weight[weight < 1 / 255] = 0
    stage('candidate')
    sr_hit = None; sr_identity = None; candidate_input = rgb
    if uses_sr(denoiser):
        sr_config = {key: value for key, value in denoiser_config.items() if key != 'method'}
        sr_identity = {'source_sha256': input_hash, 'device': device, 'denoiser_config': sr_config,
                       'precision': 'uint8-native-sr-cache', 'size': list(rgb.shape)}
        def validate_sr(value):
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != rgb.shape:
                raise ValueError('SR stage must return uint8 RGB on the original grid')
        candidate_input, sr_hit = _cached(output, 'sr', sr_identity,
            lambda: super_resolve(rgb, device=device, sr_fn=sr_fn, method=denoiser,
                                  downscale=downscale), validate_sr, warnings, inputs)
    candidate_identity = {'source_sha256': input_hash,
        'lambda': float(lam) if uses_l0(denoiser) else None, 'device': device,
        'denoiser': denoiser, 'denoiser_config': denoiser_config,
        'code': _code_hash(['../denoise.py', '../l0_cuda.py', 'fusion_pipeline.py', 'denoisers.py']),
        'callable': l0_implementation, 'precision': 'float32-cache'}
    candidate_stage = 'l0' if denoiser == 'l0' else 'candidate'

    def candidate_value():
        if uses_l0(denoiser):
            return smooth_candidate(candidate_input, float(lam), device=device, l0_fn=l0_fn)
        return candidate_input.astype(np.float32) / 255

    def validate_l0(value):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != rgb.shape
                or not np.isfinite(value).all() or np.any((value < 0) | (value > 1))):
            raise ValueError('Candidate must be finite float32 RGB in 0..1')

    candidate, l0_hit = _cached(output, candidate_stage, candidate_identity, candidate_value, validate_l0, warnings, inputs)
    if sr_model_path is not None and sr_model_path.is_file():
        # The first SR call may have fetched a missing model. Bind it before
        # exporting, just as with weights already present at invocation time.
        model_digest = _sha(sr_model_path)
        if model_digest != denoiser_config['model']['expected_sha256']:
            raise ValueError('Denoiser model changed while processing')
        input_snapshots.setdefault(str(sr_model_path), model_digest)
    stage('compose')
    restored = gradient_preserving_candidate(rgb, candidate, effective_sigma,
                                             restoration_gain=restoration_gain)
    result = blend(rgb, restored, weight)
    if not np.array_equal(result[weight == 0], rgb[weight == 0]):
        raise AssertionError('Protected or W=0 pixels changed')
    stage('encoding')
    folder.mkdir(parents=True, exist_ok=True)
    _save_rgb(folder / 'original.png', rgb, original_alpha, icc)
    _save_rgb(folder / 'result.png', result, original_alpha, icc)
    np.save(folder / 'weight.npy', weight, allow_pickle=False)
    Image.fromarray(np.ceil(weight * 255).astype(np.uint8)).save(folder / 'weight.png')
    for name in ('person_labels', 'raw_labels', 'source_map'):
        Image.fromarray(regions[name]).save(folder / (name + '.png'))
    for name, mask in [('objects', regions['objects']), ('suspected_objects', suspected_objects),
                       ('unknown', unknown),
                       ('conflict', regions['conflict']), ('consensus', regions['consensus']),
                       ('protected', regions['protected']),
                       *[('module_' + name, regions['person_labels'] == number) for name, number in MODULES.items()]]:
        Image.fromarray(mask.astype(np.uint8) * 255).save(folder / (name + '.png'))
    preview = Image.fromarray(result); preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
    preview.save(folder / 'preview.png')
    diagnostics = {key: value for key, value in regions.items() if not isinstance(value, np.ndarray) and not key.startswith('_')}
    diagnostics.update(modules=MODULES, unknown_label=255,
        note='Unknown predictions remain protected and count as incomplete. Provisional policy awaits reviewed references.')
    _json(folder / 'regions.json', diagnostics)
    if any(_sha(path) != digest for path, digest in input_snapshots.items()):
        raise RuntimeError('Source, edits or policy changed while processing; result not marked complete')
    change = np.any(result != rgb, axis=2)
    report = {'id': str(case_id), 'folder': str(case_id), 'status': 'complete', 'profile': profile,
        'source': str(source), 'source_sha256': input_hash, 'size': [rgb.shape[1], rgb.shape[0]],
        'fingerprint': fingerprint, 'parameters': parameters, 'split': split, 'resumed': False,
        'effective_sigma': effective_sigma, 'provenance': evidence['provenance'],
        'sam_records': regions['sam_records'], 'edits': edits['hashes'], 'input_paths': [str(p) for p in inputs],
        'metrics': {'active_percent': float((weight > 0).mean() * 100),
                    'unknown_percent': float(unknown.mean() * 100), 'objects_percent': float(regions['objects'].mean() * 100),
                    'suspected_objects_percent': float(suspected_objects.mean() * 100),
                    'changed_pixels': int(change.sum()), 'protected_changed_pixels': 0},
        'cache_hits': {'evidence': evidence_hit, 'l0': l0_hit, 'candidate': l0_hit, 'sr': sr_hit, 'sam': candidate_models.hits},
        'candidate_cache_identity': candidate_identity, 'candidate_cache_stage': candidate_stage,
        'sr_cache_identity': sr_identity,
        'sam_cache_misses': candidate_models.misses, 'warnings': warnings,
        'source_unchanged': True, 'no_neural_training': True, 'quality_accepted': False,
        'review_status': 'needs_review' if unknown.any() or suspected_objects.any() else 'ready',
        'quality_status': 'pending_reference_review' if regions.get('policy', {}).get('provisional', True) else 'not_evaluated',
        'elapsed_seconds': time.perf_counter() - started,
        'output_hashes': {name: _sha(folder / name) for name in OUTPUT_FILES}}
    _json(folder / 'report.json', report)
    return report


def write_records(output_root, records, *, input_paths=()):
    """Merge completed/error records and publish a compact machine-readable summary."""
    output = Path(output_root).resolve()
    inputs = [*map(Path, input_paths), *[Path(path) for record in records for path in record.get('input_paths', [])],
              *[Path(record['source']) for record in records if record.get('source')]]
    targets = [output / name for name in ('records.json', 'records.json.tmp', 'summary.csv', 'summary.csv.tmp')]
    _assert_inputs_safe(targets, inputs)
    path = output / 'records.json'
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    if not isinstance(old, list):
        raise ValueError('records.json must be a list')
    merged = {str(record['id']): record for record in old}
    merged.update({str(record['id']): record for record in records})
    values = list(merged.values()); output.mkdir(parents=True, exist_ok=True)
    _json(path, values)
    temporary = output / 'summary.csv.tmp'
    fields = ['id', 'status', 'profile', 'split', 'source_sha256', 'active_percent', 'unknown_percent', 'changed_pixels', 'error']
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for record in values:
            row = {key: record.get(key, '') for key in fields}
            row.update({key: record.get('metrics', {}).get(key, '') for key in ('active_percent', 'unknown_percent', 'changed_pixels')})
            writer.writerow(row)
    temporary.replace(output / 'summary.csv')


def _directory_manifest(directory, output):
    from .dataset import IMAGE_SUFFIXES, _groups, _inventory, _mark_prior_seen, _split
    source = Path(directory).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.is_relative_to(source):
        raise ValueError('Batch output must be outside the read-only source directory')
    files = sorted((path for path in source.rglob('*') if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                   key=lambda path: path.relative_to(source).as_posix().casefold())
    if not files:
        raise ValueError('No supported images in directory')
    if any(not path.resolve().is_relative_to(source) for path in files):
        raise ValueError('Image symlink points outside the source directory')
    collected = [_inventory(path, source, index) for index, path in enumerate(files, 1)]
    items = [item for item, _ in collected]
    groups, evidence = _groups(items, [thumb for _, thumb in collected])
    _mark_prior_seen(items, groups, []); _split(items, groups, 20261107)
    return {'items': items, 'groups': groups, 'grouping_evidence': evidence}


def process_batch(manifest=None, output_root=None, models=None, *, directory=None,
                  edits_dir=None, profile='fusion', split='all', limit=None, lam=None,
                  sigma=12, strength=1, policy=None, policy_path=None, l0_fn=None,
                  denoiser='l0', sr_fn=None, downscale=.6, structure_protection=None,
                  restoration_gain=None):
    """Process directory OR frozen manifest, pairing edits by explicit ID and SHA."""
    if (manifest is None) == (directory is None):
        raise ValueError('Specify exactly one directory or manifest')
    if profile not in ('fusion', 'reference') or split not in ('tune', 'holdout', 'all'):
        raise ValueError('Invalid profile or split')
    structure_protection, restoration_gain = _finish_settings(profile, structure_protection, restoration_gain)
    denoiser, lam = resolve_denoiser(denoiser, lam)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be positive')
    _, policy_path, policy_hash = _load_policy(policy, policy_path)
    output = Path(output_root).resolve()
    manifest_path = None
    if directory is not None:
        manifest = _directory_manifest(directory, output)
    elif not isinstance(manifest, dict):
        manifest_path = Path(manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    items = [dict(item) for item in manifest['items'] if item.get('status') == 'eligible'
             and (split == 'all' or item.get('split') == split)]
    if limit is not None:
        items = items[:limit]
    if len({str(item['id']) for item in items}) != len(items):
        raise ValueError('Duplicate case IDs in batch')
    edits_by_id = {}; edit_input_paths = []
    if edits_dir is not None:
        edit_root = Path(edits_dir).resolve()
        if not edit_root.is_dir():
            raise NotADirectoryError(edit_root)
        for path in sorted(edit_root.glob('*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or not isinstance(data.get('id'), str) or not data['id']:
                raise ValueError('Batch edits must contain explicit id')
            if data['id'] in edits_by_id:
                raise ValueError('Duplicate batch edit id: ' + data['id'])
            edits_by_id[data['id']] = path
            edit_input_paths.append(path)
            for key in ('object_add_mask', 'object_remove_mask', 'allow_mask', 'protect_mask'):
                if isinstance(data.get(key), str):
                    edit_input_paths.append((path.parent / data[key]).resolve())
            if isinstance(data.get('module_overrides'), dict):
                edit_input_paths.extend((path.parent / value).resolve() for value in data['module_overrides'].values() if isinstance(value, str))
    all_inputs = [Path(item['path']).resolve() for item in items] + edit_input_paths
    if manifest_path is not None:
        all_inputs.append(manifest_path)
    if policy_path is not None:
        all_inputs.append(policy_path)
    if uses_sr(denoiser) and sr_fn is None:
        configuration = describe_denoiser(denoiser, str(getattr(models, 'device', 'cpu')),
                                         downscale=downscale)
        all_inputs.append(Path(configuration['model']['path']).resolve())
    for item in items:
        _assert_inputs_safe(_targets(output, _case_folder(output, item['id'])), all_inputs)
    _assert_inputs_safe([output / name for name in ('records.json', 'records.json.tmp', 'summary.csv', 'summary.csv.tmp')], all_inputs)
    records = []
    for item in items:
        case_id = str(item['id']); folder = _case_folder(output, case_id)
        try:
            source = Path(item['path']).resolve()
            if item.get('sha256') and _sha(source) != item['sha256']:
                raise ValueError('Input hash differs from frozen dataset manifest')
            edit = edits_by_id.get(case_id)
            if edit is not None:
                data = json.loads(edit.read_text(encoding='utf-8'))
                if data.get('source_sha256') != _sha(source):
                    raise ValueError('Batch edits ID matched but source_sha256 differs')
            record = process_image(source, output, models, profile=profile, case_id=case_id,
                lam=lam, sigma=sigma, strength=strength, edits_path=edit, split=item.get('split', ''),
                policy=policy, policy_path=policy_path, l0_fn=l0_fn, denoiser=denoiser, sr_fn=sr_fn,
                downscale=downscale, structure_protection=structure_protection,
                restoration_gain=restoration_gain, _input_paths=all_inputs)
        except Exception as error:
            record = {'id': case_id, 'folder': case_id, 'status': 'error', 'profile': profile,
                'split': item.get('split', ''), 'source': str(item['path']),
                'input_paths': [str(path) for path in all_inputs],
                'error': type(error).__name__ + ': ' + str(error), 'no_neural_training': True}
            _assert_inputs_safe([folder / 'report.json', folder / 'report.json.tmp'], all_inputs)
            folder.mkdir(parents=True, exist_ok=True); _json(folder / 'report.json', record)
        records.append(record)
        write_records(output, records, input_paths=all_inputs)
    if not items:
        write_records(output, [], input_paths=all_inputs)
    if policy_path is not None and _sha(policy_path) != policy_hash:
        raise RuntimeError('Policy changed while processing batch')
    return records
