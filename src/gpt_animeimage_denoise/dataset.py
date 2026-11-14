"""Read-only image inventory and conservative, group-isolated evaluation splits.

This is data preparation, not supervised model training. A perceptual match is
only a possible near-duplicate; the grouping evidence is kept for inspection.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'}
DHASH_DISTANCE = 6
THUMBNAIL_MAE = 12.0  # RGB levels on a 0..255 16x16 thumbnail.
ASPECT_RELATIVE_DELTA = .03
PROCESSED_PATTERN = re.compile(
    r'(?:^|[_\- .])(toonout|denois(?:e|ed)|cleaner|esrgan|cugan|apisr|processed|upscal(?:e|ed))(?:$|[_\- .\d])',
    re.IGNORECASE,
)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _dhash(image):
    gray = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    bits = gray[:, 1:] > gray[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return f'{value:016x}'


def _family(stem):
    # A shared generation/task id is useful provenance even when two exports
    # are not close enough for a perceptual match. Grouping it is conservative.
    task = re.match(r'^(task-[a-z0-9]+)', stem, flags=re.IGNORECASE)
    if task:
        return task.group(1).casefold()
    return PROCESSED_PATTERN.split(stem, maxsplit=1)[0].rstrip('_-. ').casefold()


def _inventory(path, source, index):
    checksum = _sha(path)
    stat = path.stat()
    with Image.open(path) as opened:
        opened.load()
        original_mode = opened.mode
        original_size = list(opened.size)
        oriented = ImageOps.exif_transpose(opened)
        rgba = oriented.convert('RGBA')
        pixel_hash = hashlib.sha256(
            f'RGBA:{rgba.width}x{rgba.height}:'.encode('ascii') + rgba.tobytes()
        ).hexdigest()
        rgb = Image.new('RGB', rgba.size, 'white')
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        thumb = np.asarray(rgb.resize((16, 16), Image.Resampling.LANCZOS), dtype=np.float32)
        processed = bool(PROCESSED_PATTERN.search(path.stem))
        item = {
            'id': f'img_{index:04d}', 'index': index,
            'path': str(path.resolve()), 'relative_path': path.relative_to(source).as_posix(),
            'sha256': checksum, 'file_sha256': checksum, 'pixel_sha256': pixel_hash,
            'width': rgba.width, 'height': rgba.height, 'mode': original_mode,
            'stored_size': original_size, 'display_orientation_applied': opened.size != oriented.size,
            'dhash': _dhash(rgb), 'bytes': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'name_family': _family(path.stem),
            'status': 'excluded_from_calibration' if processed else 'eligible',
            'exclusion_reason': 'filename indicates an already processed derivative; not clean ground truth' if processed else None,
        }
    return item, thumb


def _groups(items, thumbnails):
    parents = list(range(len(items)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    evidence = []
    for right, item in enumerate(items):
        for left in range(right):
            other = items[left]
            distance = (int(item['dhash'], 16) ^ int(other['dhash'], 16)).bit_count()
            ratio_a = item['width'] / item['height']
            ratio_b = other['width'] / other['height']
            aspect_delta = abs(ratio_a - ratio_b) / max(ratio_a, ratio_b)
            mae = float(np.abs(thumbnails[right] - thumbnails[left]).mean())
            kind = None
            if item['pixel_sha256'] == other['pixel_sha256']:
                kind = 'decoded_pixel_duplicate'
            elif distance <= DHASH_DISTANCE and mae <= THUMBNAIL_MAE and aspect_delta <= ASPECT_RELATIVE_DELTA:
                kind = 'possible_near_duplicate'
            elif item['name_family'] and item['name_family'] == other['name_family']:
                kind = 'shared_name_family'
            if kind:
                parents[find(right)] = find(left)
                evidence.append({'a': other['id'], 'b': item['id'], 'kind': kind,
                    'dhash_hamming_distance': distance, 'thumbnail_rgb_mae': round(mae, 5),
                    'aspect_relative_delta': round(aspect_delta, 6)})
    grouped = {}
    for index in range(len(items)):
        grouped.setdefault(find(index), []).append(index)
    groups = []
    for indices in grouped.values():
        identity = '\n'.join(sorted({items[i]['pixel_sha256'] for i in indices}))
        group_id = 'group_' + hashlib.sha256(identity.encode('ascii')).hexdigest()[:16]
        for index in indices:
            items[index]['group_id'] = group_id
        groups.append({'group_id': group_id, 'item_ids': [items[i]['id'] for i in indices],
            'eligible_count': sum(items[i]['status'] == 'eligible' for i in indices)})
    return sorted(groups, key=lambda group: group['group_id']), evidence


def _mark_prior_seen(items, groups, names):
    if isinstance(names, (str, Path)):
        raise TypeError('prior_seen_names must be a list or tuple of task prefixes or filenames')
    requested = list(dict.fromkeys(str(name) for name in names))
    if any(not name.strip() for name in requested):
        raise ValueError('prior_seen_names cannot contain an empty name')
    families = {name: _family(Path(name).stem) for name in requested}
    matched_families = set()
    for item in items:
        item['prior_seen_direct_match'] = item['name_family'] in families.values()
        if item['prior_seen_direct_match']:
            matched_families.add(item['name_family'])
    seen_groups = {item['group_id'] for item in items if item['prior_seen_direct_match']}
    for group in groups:
        group['prior_seen'] = group['group_id'] in seen_groups
        group['forced_tune'] = group['prior_seen']
    for item in items:
        # A near duplicate/provenance sibling of a seen image cannot become an
        # unseen holdout merely because this particular filename was not listed.
        item['prior_seen'] = item['group_id'] in seen_groups
        item['forced_tune'] = item['prior_seen']
    unmatched = [name for name, family in families.items() if family not in matched_families]
    return requested, unmatched


def _split(items, groups, seed):
    eligible_groups = [group for group in groups if group['eligible_count']]
    eligible_groups.sort(key=lambda group: hashlib.sha256(
        f"{seed}:{group['group_id']}".encode('ascii')).hexdigest())
    # Guarantee both sets when independent groups exist, while keeping every
    # provenance/near-duplicate group whole. Ratios are approximate, not forced.
    count = len(eligible_groups)
    holdout_count = max(1, min(count - 1, round(count * .3))) if count >= 2 else 0
    unseen_groups = [group for group in eligible_groups if not group['forced_tune']]
    holdout = {group['group_id'] for group in unseen_groups[:holdout_count]}
    for group in groups:
        group['split'] = 'holdout' if group['group_id'] in holdout else 'tune'
    assignments = {group['group_id']: group['split'] for group in groups}
    for item in items:
        item['split'] = assignments[item['group_id']]


def _font(size):
    for candidate in [Path('C:/Windows/Fonts/msyh.ttc'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default(size=size)


def _contact_sheets(items, output):
    sheets = []
    label_font = _font(14)
    title_font = _font(22)
    colours = {'tune': '#206b4e', 'holdout': '#895724'}
    for page, start in enumerate(range(0, len(items), 20), 1):
        current = items[start:start + 20]
        rows = (len(current) + 4) // 5
        canvas = Image.new('RGB', (1500, 56 + rows * 335), '#eef1f4')
        draw = ImageDraw.Draw(canvas)
        draw.text((15, 12), f'数据素材审计 · 第 {page} 页 · 非干净真值配对', font=title_font, fill='#253144')
        for offset, item in enumerate(current):
            x = (offset % 5) * 300 + 10
            y = (offset // 5) * 335 + 56
            with Image.open(item['path']) as image:
                view = ImageOps.exif_transpose(image).convert('RGBA')
                view.thumbnail((280, 260), Image.Resampling.LANCZOS)
                preview = Image.new('RGB', view.size, 'white')
                preview.paste(view, mask=view.getchannel('A'))
                canvas.paste(preview, (x + (280 - view.width) // 2, y))
            status = item['split'] if item['status'] == 'eligible' else item['split'] + ' / 已处理，排除调优'
            if item['prior_seen'] and item['status'] == 'eligible':
                status += ' / 已见，固定调优'
            draw.text((x, y + 263), f"{item['id']} · {status}", font=label_font,
                fill=colours[item['split']] if item['status'] == 'eligible' else '#a23036')
            name = item['relative_path']
            for line_index, text in enumerate([name[:37], name[37:74]]):
                draw.text((x, y + 285 + line_index * 18), text, font=label_font, fill='#253144')
        path = output / f'contact_sheet_{page:02d}.png'
        canvas.save(path)
        sheets.append(str(path.resolve()))
    return sheets


def prepare_dataset(source: Path, output: Path, seed: int = 20261107,
                    prior_seen_names: tuple[str, ...] | list[str] = ()) -> dict:
    """Inventory inputs without modifying them; keep near-duplicate groups isolated.

    Consumers MUST filter ``status == 'eligible'`` for tuning/evaluation. Excluded
    derivatives retain their group's split, but are never clean target labels.
    Task prefixes or filenames in ``prior_seen_names`` force their entire group
    to tune. The default is empty; callers must declare their own prior usage.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if output.is_relative_to(source):
        raise ValueError('Audit output must be outside the read-only source directory')
    files = sorted((path for path in source.rglob('*') if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES), key=lambda path: path.relative_to(source).as_posix().casefold())
    if not files:
        raise ValueError(f'No supported images in {source}')
    if any(not path.resolve().is_relative_to(source) for path in files):
        raise ValueError('Image symlink points outside the source directory')
    collected = [_inventory(path, source, index) for index, path in enumerate(files, 1)]
    items = [item for item, _ in collected]
    thumbnails = [thumb for _, thumb in collected]
    groups, evidence = _groups(items, thumbnails)
    prior_seen_names, unmatched_prior_seen_names = _mark_prior_seen(items, groups, prior_seen_names)
    _split(items, groups, seed)
    output.mkdir(parents=True, exist_ok=True)
    contact_sheets = _contact_sheets(items, output)
    for item in items:
        path = Path(item['path'])
        if _sha(path) != item['sha256'] or path.stat().st_mtime_ns != item['source_mtime_ns']:
            raise RuntimeError(f'Source changed during audit: {path}')
    eligible = [item for item in items if item['status'] == 'eligible']
    counts = Counter(item['split'] for item in eligible)
    largest = max(items, key=lambda item: item['width'] * item['height'])
    annotation_candidates = sorted(str(path.relative_to(source).as_posix()) for path in source.rglob('*')
        if path.is_file() and path.suffix.lower() in {'.json', '.csv', '.jsonl', '.xml'})
    warnings = []
    if len({item['group_id'] for item in eligible}) < 2:
        warnings.append('Fewer than two eligible independent groups; no holdout set can be formed.')
    elif not counts['holdout']:
        warnings.append('All eligible groups were previously seen; no unseen holdout set can be formed.')
    if unmatched_prior_seen_names:
        warnings.append('Some prior_seen_names matched no input family; inspect unmatched_prior_seen_names.')
    report = {
        'schema_version': 1, 'source': str(source), 'seed': seed,
        'prior_seen_names': prior_seen_names,
        'unmatched_prior_seen_names': unmatched_prior_seen_names,
        'items': items, 'groups': groups, 'grouping_evidence': evidence,
        'grouping_policy': {'dhash_bits': 64, 'max_hamming_distance': DHASH_DISTANCE,
            'thumbnail_rgb_mae_max': THUMBNAIL_MAE, 'thumbnail_size': [16, 16],
            'aspect_relative_delta_max': ASPECT_RELATIVE_DELTA,
            'shared_generation_name_family_also_grouped': True,
            'note': 'Perceptual matches are possible near-duplicates, not proof of identity; connected components are kept in one split.'},
        'split_policy': {'target_tune_fraction': .7, 'unit': 'independent group',
            'excluded_items_keep_group_split': True, 'eligibility_filter': "status == 'eligible'",
            'forced_tune_groups': [group['group_id'] for group in groups if group['forced_tune']],
            'prior_seen_note': 'Declared previously reviewed inputs and their whole groups are forced to tune; this is not inferred from filenames alone.'},
        'supervision': {'verified_clean_reference_pairs': 0, 'verified_mask_labels': 0,
            'annotation_candidates': annotation_candidates,
            'note': 'This inventory does not establish supervised labels. Processed derivatives are not clean ground truth. No neural model has been trained by this operation.'},
        'summary': {'total_images': len(items), 'eligible_images': len(eligible),
            'excluded_from_calibration': len(items) - len(eligible), 'independent_groups': len(groups),
            'eligible_groups': sum(bool(group['eligible_count']) for group in groups),
            'tune_images': counts['tune'], 'holdout_images': counts['holdout'],
            'prior_seen_images': sum(item['prior_seen'] for item in items),
            'forced_tune_groups': sum(group['forced_tune'] for group in groups),
            'grouping_edges': dict(Counter(edge['kind'] for edge in evidence)),
            'largest_image': {'id': largest['id'], 'width': largest['width'], 'height': largest['height'],
                'pixels': largest['width'] * largest['height']},
            'original_files_unchanged': True,
            'warnings': warnings},
        'contact_sheets': contact_sheets,
    }
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'summary.json').write_text(json.dumps(report['summary'], ensure_ascii=False, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--prior-seen', action='append', default=[], metavar='NAME',
        help='Task prefix or filename already reviewed; repeat for each known sample family')
    args = parser.parse_args()
    result = prepare_dataset(args.source, args.output, args.seed, prior_seen_names=args.prior_seen)
    print(json.dumps({'summary': result['summary'], 'contact_sheets': result['contact_sheets']}, ensure_ascii=False, indent=2))
