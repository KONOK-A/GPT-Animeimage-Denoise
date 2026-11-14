import importlib
from pathlib import Path
import tempfile
import unittest

from PIL import Image


class AppendBatchInputs(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('tools.batch_acceptance.append_comparison')
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.image=self.root/'clipboard.png';Image.new('RGB',(13,17),(80,120,160)).save(self.image)

    def test_attachment_is_copied_without_reencoding_and_assigned_next_id(self):
        items=self.module.snapshot_inputs([self.image],self.root/'extension',7)
        self.assertEqual(items[0]['id'],'batch_0007')
        self.assertEqual(Path(items[0]['path']).read_bytes(),self.image.read_bytes())
        self.assertEqual(items[0]['original_temp_path'],str(self.image.resolve()))
        self.assertEqual((items[0]['width'],items[0]['height']),(13,17))

    def test_existing_snapshot_is_not_silently_overwritten(self):
        out=self.root/'extension';first=self.module.snapshot_inputs([self.image],out,7)
        before=Path(first[0]['path']).read_bytes()
        Image.new('RGB',(13,17),(1,2,3)).save(self.image)
        with self.assertRaises(ValueError):self.module.snapshot_inputs([self.image],out,7)
        self.assertEqual(Path(first[0]['path']).read_bytes(),before)


if __name__=='__main__':unittest.main()
