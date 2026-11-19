from contextlib import redirect_stdout
from io import BytesIO, StringIO, TextIOWrapper
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from gpt_animeimage_denoise import __version__, cli


class FusionCLITest(unittest.TestCase):
    def test_version_works_without_a_command_or_models(self):
        output = StringIO()
        modules = {'gpt_animeimage_denoise.fusion_models': None, 'torch': None}
        with (mock.patch.dict(sys.modules, modules),
              mock.patch.object(sys, 'argv', ['gaid']),
              redirect_stdout(output)):
            with self.assertRaises(SystemExit) as result:
                cli.main(['--version'])

        self.assertEqual(result.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), f'gaid {__version__}')

    def test_model_help_supports_redirected_gbk_output(self):
        models = Path('models-\N{TEST TUBE}').resolve()
        environment = {'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR': str(models)}
        with mock.patch.dict('os.environ', environment):
            for command in ('process', 'batch'):
                with (self.subTest(command=command),
                      TextIOWrapper(BytesIO(), encoding='gbk') as output,
                      redirect_stdout(output)):
                    with self.assertRaises(SystemExit) as result:
                        cli.main([command, '--help'])
                    self.assertEqual(result.exception.code, 0)

    def test_preview_dispatches_without_importing_torch_or_models(self):
        preview = mock.Mock(return_value={'output': 'preview.png'})
        modules = {'gpt_animeimage_denoise.preview': SimpleNamespace(preview_case=preview),
                   'gpt_animeimage_denoise.fusion_models': None, 'torch': None}
        with mock.patch.dict(sys.modules, modules):
            code = cli.main(['preview', 'case', '--out', 'preview.png',
                             '--strength', '.5', '--restoration-gain', '.35', '--sigma', '36'])

        self.assertEqual(code, 0)
        preview.assert_called_once_with(Path('case'), Path('preview.png'),
                                        strength=.5, restoration_gain=.35, sigma=36.)

    def test_retired_legacy_profile_is_rejected_by_the_parser(self):
        with self.assertRaises(SystemExit) as context:
            cli.main(['process', 'a.png', '--out', 'out', '--profile', 'legacy'])
        self.assertEqual(context.exception.code, 2)

    def test_retired_report_command_is_rejected_by_the_parser(self):
        with self.assertRaises(SystemExit) as context:
            cli.main(['report', '--help'])
        self.assertEqual(context.exception.code, 2)

    def test_directory_and_manifest_are_exclusive(self):
        with self.assertRaises(SystemExit) as context:
            cli.main(['batch', '--directory', 'x', '--manifest', 'y', '--out', 'z'])
        self.assertEqual(context.exception.code, 2)

    def test_invalid_strength_fails_before_model_import(self):
        with self.assertRaises(SystemExit) as context:
            cli.main(['process', 'a.png', '--out', 'out', '--strength', '2'])
        self.assertEqual(context.exception.code, 2)

    def test_fusion_cli_writes_records_and_releases_models(self):
        model = mock.Mock(device='cpu', signature='test')
        report = dict(id='x', folder='x', status='complete', resumed=False)
        asset_root = Path('external-models').resolve()
        with mock.patch.dict('os.environ', {'GPT_ANIMEIMAGE_DENOISE_MODEL_DIR': str(asset_root)}), \
             mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels', return_value=model) as factory, \
             mock.patch('gpt_animeimage_denoise.fusion_pipeline.process_image', return_value=report) as process, \
             mock.patch('gpt_animeimage_denoise.fusion_pipeline.write_records') as write:
            code = cli.main(['process', 'a.png', '--out', 'out',
                             '--strength', '.6', '--device', 'cpu', '--policy', 'calibrated.json',
                             '--structure-protection', '.5', '--restoration-gain', '1'])
        self.assertEqual(code, 0)
        factory.assert_called_once_with(asset_root / 'v2', 'cpu', anime_assets=asset_root / 'AnimeSeg')
        self.assertEqual(process.call_args.kwargs['strength'], .6)
        self.assertEqual(process.call_args.kwargs['profile'], 'fusion')
        self.assertEqual(process.call_args.kwargs['denoiser'], 'l0')
        self.assertEqual(process.call_args.kwargs['structure_protection'], .5)
        self.assertEqual(process.call_args.kwargs['restoration_gain'], 1)
        self.assertEqual(process.call_args.kwargs['policy_path'], Path('calibrated.json'))
        write.assert_called_once_with(Path('out'), [report], input_paths=[Path('a.png'), Path('calibrated.json')])
        model.release.assert_called_once()

    def test_fusion_batch_propagates_directory_edits_split_and_releases_on_error(self):
        model = mock.Mock(device='cpu', signature='test')
        with mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels', return_value=model), \
             mock.patch('gpt_animeimage_denoise.fusion_pipeline.process_batch', side_effect=ValueError('bad input')) as process:
            code = cli.main(['batch', '--directory', 'inputs', '--edits-dir', 'edits',
                '--out', 'out', '--profile', 'reference', '--split', 'tune', '--device', 'cpu', '--policy', 'calibrated.json'])
        self.assertEqual(code, 1)
        self.assertEqual(process.call_args.kwargs['directory'], Path('inputs'))
        self.assertEqual(process.call_args.kwargs['edits_dir'], Path('edits'))
        self.assertEqual(process.call_args.kwargs['split'], 'tune')
        self.assertEqual(process.call_args.kwargs['profile'], 'reference')
        self.assertEqual(process.call_args.kwargs['policy_path'], Path('calibrated.json'))
        model.release.assert_called_once()

    def test_l0_can_be_selected_for_reference_comparison(self):
        model=mock.Mock(device='cpu',signature='test')
        record=dict(id='x',folder='x',status='complete',resumed=False)
        with mock.patch('gpt_animeimage_denoise.fusion_models.FusionModels',return_value=model), \
             mock.patch('gpt_animeimage_denoise.fusion_pipeline.process_image',return_value=record) as process, \
             mock.patch('gpt_animeimage_denoise.fusion_pipeline.write_records'):
            result=cli.main(['process','a.png','--out','out','--profile','reference','--denoiser','l0','--device','cpu'])
        self.assertEqual(result,0);self.assertEqual(process.call_args.kwargs['denoiser'],'l0')
