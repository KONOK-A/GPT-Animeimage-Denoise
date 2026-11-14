import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


class BatchRunChecks(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('tools.batch_acceptance.run_comparison')
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'input.png'
        self.rgb=np.full((8,10,3),100,np.uint8)
        Image.fromarray(self.rgb).save(self.source)

    def pair(self):
        reports={}
        for method in ('l0','esrgan-l0'):
            folder=self.root/method;folder.mkdir()
            labels=np.full((8,10),6,np.uint8);labels[2:6,2:7]=3
            weight=np.zeros((8,10),np.float32);weight[3:5,3:6]=.5
            result=self.rgb.copy();result[weight>0]=110
            for name,array in [('person_labels',labels),('raw_labels',labels),
                ('objects',np.zeros_like(labels)),('unknown',np.zeros_like(labels)),
                ('conflict',np.zeros_like(labels)),('consensus',np.zeros_like(labels)),
                ('protected',np.where(weight==0,255,0).astype(np.uint8)),('source_map',labels)]:
                Image.fromarray(array).save(folder/(name+'.png'))
            for number,name in enumerate(('skin','face','hair','eyes','mouth','outfit'),1):
                Image.fromarray((labels==number).astype(np.uint8)*255).save(folder/('module_'+name+'.png'))
            Image.fromarray(self.rgb).save(folder/'original.png');Image.fromarray(result).save(folder/'result.png')
            np.save(folder/'weight.npy',weight,allow_pickle=False)
            p={'profile':'fusion','lambda':.0005,'sigma_at_2048':12.,'strength':1.,'device':'cpu',
                'models':'same','implementation':'same','policy':{'see_reference':'selective'},'denoiser':method}
            report={'status':'complete','source_sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),
                'folder':method,'parameters':p,'output_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.iterdir()}}
            reports[method]=report
        return reports

    def test_pair_checks_actual_masks_weights_source_and_protected_pixels(self):
        result=self.module.verify_pair(self.source,self.root,self.pair())
        self.assertTrue(result['same_masks'] and result['same_weight'] and result['protected_unchanged'])
        self.assertEqual(result['alpha'],'not_present');self.assertEqual(result['icc'],'not_present')

    def test_mask_difference_rejected_even_when_report_hash_is_updated(self):
        reports=self.pair();path=self.root/'esrgan-l0/module_hair.png'
        Image.new('L',(10,8),0).save(path)
        reports['esrgan-l0']['output_hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):self.module.verify_pair(self.source,self.root,reports)

    def test_zero_weight_change_rejected(self):
        reports=self.pair();path=self.root/'l0/result.png'
        bad=self.rgb.copy();bad[0,0]=101;Image.fromarray(bad).save(path)
        reports['l0']['output_hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):self.module.verify_pair(self.source,self.root,reports)


if __name__=='__main__':unittest.main()
