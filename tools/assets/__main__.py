"""Verify, import or fetch the locked local model assets for V2.1."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


MANIFEST = Path(__file__).resolve().parents[2] / 'src/gpt_animeimage_denoise/model_manifest.json'
ENV_MODEL_DIR = 'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR'
LABELS = ('background', 'skin', 'face', 'hair_main', 'left_eye', 'right_eye',
          'left_eyebrow', 'right_eyebrow', 'nose', 'mouth', 'clothes', 'accessory')


def asset_path(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Asset path escapes its model directory: {name}')
    return path


def check_hash(path, expected, name):
    if not path.is_file():
        raise FileNotFoundError(f'Missing required asset: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected:
        raise ValueError(f'SHA-256 mismatch for {name}: expected {expected}, got {digest}')


def verify(root, manifest):
    root = root.resolve()
    count = 0
    for name, item in manifest.items():
        path = asset_path(root, name)
        if item.get('optional') and not path.exists():
            continue
        check_hash(path, item['sha256'], name)
        count += 1
    return count


def write_verified(root, name, expected, write):
    target = asset_path(root, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.',
                                     suffix='.part', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        write(temporary)
        check_hash(temporary, expected, name)
        # Publish the verified bytes atomically without replacing an existing file.
        try:
            os.link(temporary, target)
        except FileExistsError:
            check_hash(target, expected, name)
    finally:
        temporary.unlink()


def pending_assets(root, entries):
    pending = []
    for name, item in entries:
        target = asset_path(root, name)
        if target.exists() or target.is_symlink():
            check_hash(target, item['sha256'], name)
        else:
            pending.append((name, item))
    return pending


def import_assets(source, destination, manifest):
    source, destination = source.resolve(), destination.resolve()
    verify(source, manifest)
    entries = [(name, item) for name, item in manifest.items()
               if asset_path(source, name).is_file()]
    pending = pending_assets(destination, entries)
    for name, item in pending:
        original = asset_path(source, name)
        write_verified(destination, name, item['sha256'], lambda path: shutil.copyfile(original, path))
    return len(pending)


def download(item, destination):
    request = urllib.request.Request(item['url'], headers={'User-Agent': 'gpt-animeimage-denoise-assets'})
    with urllib.request.urlopen(request, timeout=90) as response, destination.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)


def generated_bytes(kind, root, manifest):
    if kind == 'mask2former-namespace':
        return b'# Isolated experiment namespace for unmodified upstream modules.\r\n'
    if kind == 'mask2former-label-config':
        source = asset_path(root, 'AnimeSeg/base-configs/large/upstream_config.json')
        data = json.loads(source.read_text(encoding='utf-8'))
        data['id2label'] = dict(enumerate(LABELS))
        data['label2id'] = {label: index for index, label in enumerate(LABELS)}
        data['num_labels'] = len(LABELS)
    elif kind == 'v2-checksums':
        names = ('skytnt/isnet.py', 'skytnt/model.safetensors', 'skytnt/LICENSE',
                 'see_through/checkpoint-18000.pt', 'see_through/LICENSE',
                 'sam2/sam2.1_hiera_large.pt')
        data = {name.replace('/', '\\'): manifest[f'v2/{name}']['sha256'] for name in names}
    else:
        raise ValueError(f'Unknown generated asset: {kind}')
    # The established model loader validates these legacy files byte-for-byte.
    return json.dumps(data, indent=2).replace('\n', '\r\n').encode('utf-8')


def fetch_assets(root, manifest):
    root = root.resolve()
    pending = pending_assets(root, manifest.items())
    count = 0
    for name, item in pending:
        if 'url' in item:
            print(f'Downloading {name}', flush=True)
            write_verified(root, name, item['sha256'], lambda path: download(item, path))
            count += 1
    for name, item in pending:
        if 'generate' in item:
            content = generated_bytes(item['generate'], root, manifest)
            write_verified(root, name, item['sha256'], lambda path: path.write_bytes(content))
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    default_root = Path(os.environ.get(ENV_MODEL_DIR) or 'models')
    verify_parser = commands.add_parser('verify', help='check an existing model directory')
    verify_parser.add_argument('root', type=Path, nargs='?', default=default_root)
    import_parser = commands.add_parser('import', help='copy locked assets from an existing directory')
    import_parser.add_argument('source', type=Path)
    import_parser.add_argument('--dest', type=Path, default=default_root)
    fetch_parser = commands.add_parser('fetch', help='download and verify locked segmentation assets')
    fetch_parser.add_argument('--dest', type=Path, default=default_root)
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    try:
        if args.command == 'verify':
            count = verify(args.root.expanduser(), manifest)
            print(f'Verified {count} model asset files in {args.root.resolve()}')
        elif args.command == 'import':
            count = import_assets(args.source.expanduser(), args.dest.expanduser(), manifest)
            print(f'Imported {count} model asset files into {args.dest.resolve()}')
        else:
            count = fetch_assets(args.dest.expanduser(), manifest)
            checked = verify(args.dest.expanduser(), manifest)
            print(f'Prepared {count} files and verified {checked} in {args.dest.resolve()}')
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
