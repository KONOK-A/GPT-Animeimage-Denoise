import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image,PngImagePlugin

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gpt_animeimage_denoise.dataset import prepare_dataset


class DatasetAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'训练 图片';self.source.mkdir()

    def tearDown(self):self.temp.cleanup()

    def write(self,name,pixels,metadata=None):
        path=self.source/name
        Image.fromarray(pixels).save(path,pnginfo=metadata)
        return path

    def test_decoded_duplicates_group_despite_different_png_metadata(self):
        pixels=np.random.default_rng(17).integers(0,256,(64,64,3),dtype=np.uint8)
        first=self.write('原图.png',pixels)
        meta=PngImagePlugin.PngInfo();meta.add_text('note','same pixels, different encoding')
        second=self.write('副本.png',pixels,meta)
        report=prepare_dataset(self.source,self.root/'审计')
        one,two=report['items']
        self.assertNotEqual(one['sha256'],two['sha256'])
        self.assertEqual(one['pixel_sha256'],two['pixel_sha256'])
        self.assertEqual(one['group_id'],two['group_id'])
        self.assertEqual(one['split'],two['split'])
        self.assertEqual(len(one['dhash']),16)
        self.assertEqual({one['path'],two['path']},{str(first.resolve()),str(second.resolve())})

    def test_near_variants_do_not_leak_and_split_is_deterministic(self):
        pixels=np.random.default_rng(19).integers(10,240,(96,64,3),dtype=np.uint8)
        self.write('角色.png',pixels)
        self.write('角色_ToonOut.png',pixels+2)
        for index in range(10):
            other=np.random.default_rng(index+100).integers(0,256,(64,96,3),dtype=np.uint8)
            self.write(f'独立{index:02}.png',other)
        a=prepare_dataset(self.source,self.root/'audit1',seed=22)
        b=prepare_dataset(self.source,self.root/'audit2',seed=22)
        state=lambda result:{v['relative_path']:(v['group_id'],v['split'],v['status']) for v in result['items']}
        self.assertEqual(state(a),state(b))
        items={v['relative_path']:v for v in a['items']}
        self.assertEqual(items['角色.png']['group_id'],items['角色_ToonOut.png']['group_id'])
        self.assertEqual(items['角色.png']['split'],items['角色_ToonOut.png']['split'])
        self.assertEqual(items['角色_ToonOut.png']['status'],'excluded_from_calibration')
        self.assertEqual({v['split'] for v in a['items'] if v['status']=='eligible'},{'tune','holdout'})
        for group in {v['group_id'] for v in a['items']}:
            self.assertEqual(len({v['split'] for v in a['items'] if v['group_id']==group}),1)
        self.assertFalse(a['supervision']['verified_clean_reference_pairs'])
        self.assertIn('possible_near_duplicate',{e['kind'] for e in a['grouping_evidence']})

    def test_flat_images_with_equal_dhash_but_different_colours_stay_apart(self):
        self.write('dark.png',np.full((64,64,3),20,np.uint8))
        self.write('light.png',np.full((64,64,3),220,np.uint8))
        report=prepare_dataset(self.source,self.root/'audit')
        a,b=report['items']
        self.assertEqual(a['dhash'],b['dhash'])
        self.assertNotEqual(a['group_id'],b['group_id'])

    def test_source_bytes_and_timestamp_unchanged_and_reports_readable(self):
        source=self.write('含 空格.png',np.zeros((32,48,3),np.uint8))
        before=(hashlib.sha256(source.read_bytes()).hexdigest(),source.stat().st_mtime_ns)
        result=prepare_dataset(self.source,self.root/'结果 审计')
        after=(hashlib.sha256(source.read_bytes()).hexdigest(),source.stat().st_mtime_ns)
        self.assertEqual(before,after)
        manifest=json.loads((self.root/'结果 审计/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest,result)
        self.assertEqual((result['items'][0]['width'],result['items'][0]['height']),(48,32))
        for path in result['contact_sheets']:
            with Image.open(path) as image:image.verify()

    def test_source_directory_cannot_also_be_output(self):
        self.write('image.png',np.zeros((16,16,3),np.uint8))
        with self.assertRaises(ValueError):prepare_dataset(self.source,self.source/'审计')

    def test_previously_seen_family_is_forced_to_tune_with_its_derivatives(self):
        for index in range(10):
            pixels=np.random.default_rng(300+index).integers(0,256,(64,96,3),dtype=np.uint8)
            self.write(f'task-example{index:02}.png',pixels)
            self.write(f'task-example{index:02}_ToonOut.png',pixels)
        baseline=prepare_dataset(self.source,self.root/'before',seed=72)
        prior=next(v for v in baseline['items'] if v['status']=='eligible' and v['split']=='holdout')
        prefix=Path(prior['relative_path']).stem
        result=prepare_dataset(self.source,self.root/'after',seed=72,prior_seen_names=(prefix,))
        family=[v for v in result['items'] if v['group_id']==prior['group_id']]
        self.assertEqual(len(family),2)
        self.assertTrue(all(v['prior_seen'] and v['forced_tune'] and v['split']=='tune' for v in family))
        self.assertEqual(result['summary']['forced_tune_groups'],1)
        self.assertEqual(result['prior_seen_names'],[prefix])
        self.assertTrue(any(v['split']=='holdout' for v in result['items']))
        for group in result['groups']:
            members=[v for v in result['items'] if v['group_id']==group['group_id']]
            self.assertEqual(len({v['split'] for v in members}),1)
            if group['prior_seen']:self.assertEqual(group['split'],'tune')

    def test_all_seen_groups_produce_no_false_holdout(self):
        for index in range(2):
            pixels=np.random.default_rng(400+index).integers(0,256,(64,96,3),dtype=np.uint8)
            self.write(f'task-known{index}.png',pixels)
        report=prepare_dataset(self.source,self.root/'audit',prior_seen_names=['task-known0','task-known1'])
        self.assertEqual(report['summary']['holdout_images'],0)
        self.assertEqual(report['summary']['forced_tune_groups'],2)
        self.assertTrue(report['summary']['warnings'])
        self.assertTrue(all(v['split']=='tune' and v['prior_seen'] for v in report['items']))


if __name__=='__main__':unittest.main()
