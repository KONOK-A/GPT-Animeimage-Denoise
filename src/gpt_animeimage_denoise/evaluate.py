"""Prepare review references, calibrate dev policies, evaluate frozen holdout."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare-review')
    prepare.add_argument('--manifest', type=Path, required=True)
    prepare.add_argument('--predictions', type=Path, required=True)
    prepare.add_argument('--out', type=Path, required=True)
    prepare.add_argument('--count', type=int, default=12)
    comparisons = commands.add_parser('add-comparisons')
    comparisons.add_argument('--review', type=Path, required=True)
    comparisons.add_argument('--predictions', type=Path, required=True)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--review', type=Path, required=True)
    calibration.add_argument('--out', type=Path, required=True)
    calibration.add_argument('--device', default='cuda')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--review', type=Path, required=True)
    evaluate.add_argument('--reference', type=Path, required=True)
    evaluate.add_argument('--candidate', type=Path, required=True)
    evaluate.add_argument('--policy', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)

    from .fusion_evaluation import (calibrate_policies, evaluate_release, load_reviewed_cases,
                                   prepare_review, render_denoise_review, review_input_paths,
                                   safe_evaluation_output)

    if args.command == 'add-comparisons':
        result = render_denoise_review(args.review, args.predictions)
        print(json.dumps({'cases': len(result['cases']), 'quality_accepted': False}, ensure_ascii=False))
        return 0
    if args.command == 'prepare-review':
        result = prepare_review(args.manifest, args.predictions, args.out, args.count)
        print(json.dumps({'cases': len(result['cases']), 'approval_required': True,
                          'output': str(args.out)}, ensure_ascii=False))
        return 0
    if args.command == 'evaluate':
        safe_evaluation_output(args.out, args.review, policy=args.policy,
                               prediction_roots=[args.reference, args.candidate])
        result = evaluate_release(args.review, args.reference, args.candidate, args.policy)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + '.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(args.out)
        print(json.dumps(result['gate'], ensure_ascii=False))
        return 0 if result['gate']['accepted'] else 2

    # Check user approval before constructing heavyweight model adapters.
    load_reviewed_cases(args.review, 'tune')
    import numpy as np
    from PIL import Image

    from .fusion_models import FusionModels
    from .fusion_pipeline import process_image

    models = FusionModels(device=args.device)

    def run(case, policy, profile):
        folder = args.out / 'runs' / profile
        process_image(case['source'], folder, models, case_id=case['id'], profile=profile, policy=policy,
                      split='tune', _input_paths=review_input_paths(args.review))
        return {'person_labels': np.array(Image.open(folder / case['id'] / 'person_labels.png')),
                'objects': np.array(Image.open(folder / case['id'] / 'objects.png')) > 0}

    try:
        result = calibrate_policies(args.review, lambda c, p: run(c, p, 'fusion'),
                                    lambda c: run(c, None, 'reference'), output=args.out)
    finally:
        models.release()
    print(json.dumps({'status': result['status'], 'holdout_used': False}, ensure_ascii=False))
    return 0 if result['selected_policy'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
