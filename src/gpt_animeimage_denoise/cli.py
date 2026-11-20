"""Denoise anime illustrations, preview cached results, or open the workbench."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .denoisers import METHOD_CHOICES
from .devices import DEVICE_HELP, resolve_device
from .model_paths import model_root


def _fraction(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError('must be finite in 0..1')
    return value


def _model_args(parser):
    parser.epilog = ('Set GPT_ANIMEIMAGE_DENOISE_MODEL_DIR to choose the model root. '
                     'It defaults to ./models relative to the current directory.')
    parser.add_argument('--device', default='auto', help=DEVICE_HELP)
    parser.add_argument('--assets', type=Path,
                        help='Region model directory (defaults to <model root>/v2)')
    parser.add_argument('--anime-assets', type=Path,
                        help='AnimeSeg model directory (defaults to <model root>/AnimeSeg)')
    parser.add_argument('--lambda', '--l0', dest='lam', type=float, default=None,
                        help='L0 strength; defaults to 0.0005 or the named L0 preset')
    parser.add_argument('--sigma', type=float, default=12, help='Residual sigma at short-edge 2048')
    profiles = parser.add_mutually_exclusive_group()
    profiles.add_argument('--profile', choices=['reference', 'fusion'], default='fusion',
                          help='Part-segmentation profile (default: fusion)')
    profiles.add_argument('--whole', dest='profile', action='store_const', const='whole',
                          help='Denoise the whole visible image without part segmentation')
    parser.add_argument('--strength', type=_fraction, default=1, help='Blend strength in 0..1')
    parser.add_argument('--structure-protection', type=_fraction,
                        help='Automatic line protection in 0..1 (fusion default 0.2, reference default 1)')
    parser.add_argument('--restoration-gain', type=_fraction,
                        help='Gaussian residual restoration in 0..1 (fusion default 0.25, reference default 1)')
    parser.add_argument('--policy', type=Path, help='Validated policy JSON')
    parser.add_argument('--denoiser', '--method', dest='denoiser', choices=METHOD_CHOICES, default='l0',
                        help='Candidate method (default: l0)')
    parser.add_argument('--downscale', type=float, default=.6,
                        help='Pre-scale for ESRGAN in 0..1; APISR uses the original input')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Run gaid <command> --help for command options.')
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    commands = parser.add_subparsers(dest='command', required=True, title='commands')

    inventory = commands.add_parser('inventory', help='Create a manifest for repeatable batches',
                                    description='Scan an image directory and write a dataset manifest.')
    inventory.add_argument('input', type=Path, help='Source image directory to scan recursively')
    inventory.add_argument('--out', type=Path, required=True,
                           help='Directory for manifest.json and summaries. Must be outside the source directory.')
    inventory.add_argument('--seed', type=int, default=20261107, help='Dataset split seed (defaults to %(default)s)')
    inventory.add_argument('--prior-seen', action='append', default=[],
                           help='Previously seen filename or task prefix (repeatable)')

    process = commands.add_parser('process', help='Denoise one image by part or over the whole image',
                                  description='Save result.png, weight.png, and report.json in a case directory.')
    process.add_argument('input', type=Path, help='Source image to process')
    process.add_argument('--out', type=Path, required=True, help='Output root for the case directory and cache')
    process.add_argument('--edits', type=Path, help='Corrections JSON for this source image')
    _model_args(process)

    preview = commands.add_parser('preview', help='Recombine a completed case without model inference')
    preview.add_argument('case', type=Path, help='Completed case directory containing report.json')
    preview.add_argument('--out', type=Path, required=True, help='New PNG path outside the case and its cache')
    preview.add_argument('--strength', type=_fraction, help='Absolute blend strength in 0..1')
    preview.add_argument('--restoration-gain', type=_fraction, help='Gaussian residual restoration in 0..1')
    preview.add_argument('--sigma', type=float, help='Positive residual sigma at short-edge 2048')

    ui = commands.add_parser('ui', help='Open the local browser workbench')
    ui.add_argument('--port', type=int, default=0, help='Local port (default: choose an available port)')
    ui.add_argument('--no-browser', action='store_true', help='Print the address without opening a browser')

    batch = commands.add_parser('batch', help='Denoise images from a directory or manifest',
                                description='Process eligible images and write a summary.csv with per-image results.')
    batch_input = batch.add_mutually_exclusive_group(required=True)
    batch_input.add_argument('--manifest', type=Path, help='Manifest JSON created by gaid inventory')
    batch_input.add_argument('--directory', type=Path, help='Source image directory to scan recursively')
    batch.add_argument('--edits-dir', type=Path, help='Directory of corrections JSON files matched by case ID')
    batch.add_argument('--out', type=Path, required=True,
                       help='Output root for cases and summary.csv. Keep it outside --directory.')
    batch.add_argument('--split', choices=['tune', 'holdout', 'all'], default='all',
                       help='Dataset split to process (defaults to %(default)s)')
    batch.add_argument('--limit', type=int, help='Maximum number of cases to process after filtering by split')
    _model_args(batch)

    args = parser.parse_args(argv)
    if getattr(args, 'profile', None) == 'whole':
        part_options = ['--' + name.replace('_', '-')
                        for name in ('assets', 'anime_assets', 'policy', 'edits', 'edits_dir')
                        if getattr(args, name, None) is not None]
        if part_options:
            parser.error('--whole cannot be combined with ' + ', '.join(part_options))
    try:
        if args.command == 'ui':
            from .workbench import launch_workbench

            return launch_workbench(port=args.port, open_browser=not args.no_browser)

        if args.command == 'inventory':
            from .dataset import prepare_dataset

            result = prepare_dataset(args.input, args.out, args.seed, prior_seen_names=args.prior_seen)
            print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
            return 0

        if args.command == 'preview':
            from .preview import preview_case

            result = preview_case(args.case, args.out, strength=args.strength,
                                  restoration_gain=args.restoration_gain, sigma=args.sigma)
            print(json.dumps(result, ensure_ascii=False))
            return 0

        device = resolve_device(args.device)

        from .fusion_pipeline import process_batch, process_image, write_records

        options = dict(profile=args.profile, strength=args.strength, lam=args.lam, sigma=args.sigma,
                       policy_path=args.policy, denoiser=args.denoiser, downscale=args.downscale,
                       structure_protection=args.structure_protection, restoration_gain=args.restoration_gain)
        models = None
        if args.profile == 'whole':
            options['device'] = device
        else:
            from .fusion_models import FusionModels

            assets_root = model_root()
            models = FusionModels(args.assets or assets_root / 'v2', device,
                                  anime_assets=args.anime_assets or assets_root / 'AnimeSeg')
        try:
            if args.command == 'batch':
                records = process_batch(args.manifest, args.out, models, directory=args.directory,
                                        edits_dir=args.edits_dir, split=args.split, limit=args.limit, **options)
                return 1 if any(record['status'] == 'error' for record in records) else 0

            result = process_image(args.input, args.out, models, edits_path=args.edits, **options)
            inputs = [path for path in (args.input, args.edits, args.policy) if path is not None]
            write_records(args.out, [result], input_paths=inputs)
            print(json.dumps({'id': result['id'], 'status': result['status'], 'resumed': result['resumed'],
                              'output': str(args.out / result['folder'])}, ensure_ascii=False))
            return 0
        finally:
            if models is not None:
                models.release()
    except Exception as error:
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
