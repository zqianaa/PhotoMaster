"""CPU regression tests for preparing photos and stopping invalid VLM runs."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module('prepare_input', 'scripts/inference/prepare_input.py').prepare
convert = load_module('convert', 'scripts/inference/vlm_to_dit_bench_reformat.py').convert
check_data = load_module('check_setup', 'scripts/check_setup.py').check_data


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.jpg'
        image = Image.new('RGB', (30, 20), 'blue')
        exif = Image.Exif()
        exif[274] = 6  # rotate 90 degrees on display
        exif[270] = 'private image metadata'
        image.save(self.source, exif=exif)

    def test_prepare_preserves_source_normalizes_orientation_and_removes_metadata(self):
        before = self.source.read_bytes()
        dest = prepare(self.source, 'demo_portrait', self.root)
        self.assertEqual(before, self.source.read_bytes())
        self.assertEqual(check_data(dest), [])
        with Image.open(dest / 'img/input.png') as image:
            self.assertEqual(image.size, (20, 30))
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(dict(image.getexif()), {})

    def test_prepare_refuses_overwrite(self):
        dest = prepare(self.source, 'demo_portrait', self.root)
        before = (dest / 'metadata.json').read_bytes()
        with self.assertRaises(FileExistsError):
            prepare(self.source, 'demo_portrait', self.root)
        self.assertEqual(before, (dest / 'metadata.json').read_bytes())

    def test_prepare_rejects_path_traversal_and_existing_bench_names(self):
        for name in ('../outside', 'demo_../../outside', 'photomaster_bench', 'demo_'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                prepare(self.source, name, self.root)

    def test_bad_image_leaves_no_demo(self):
        self.source.write_text('not an image')
        with self.assertRaises(OSError):
            prepare(self.source, 'demo_invalid', self.root)
        self.assertFalse((self.root / 'bench/demo_invalid').exists())

    def test_check_data_reports_missing_image_and_duplicate_output(self):
        dest = prepare(self.source, 'demo_portrait', self.root)
        records = json.loads((dest / 'metadata.json').read_text())
        records.append(dict(records[0], index=2, original_img_path='img/missing.png'))
        (dest / 'metadata.json').write_text(json.dumps(records))
        errors = check_data(dest)
        self.assertTrue(any('missing' in error for error in errors))
        self.assertTrue(any('save_name' in error for error in errors))


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'vlm.json'
        self.output = self.root / 'dit.jsonl'
        self.record = {'index': 1, 'original_img_path': 'img/photo.png',
                       'Image_Editing_Prompt': 'A medium shot with soft side light.'}

    def write(self, records):
        self.source.write_text(json.dumps({'results': records}))

    def test_success_preserves_prompt_and_resolvable_relative_image(self):
        self.write([self.record])
        convert(self.source, self.output)
        result = json.loads(self.output.read_text())
        self.assertEqual(result['caption'], [self.record['Image_Editing_Prompt']])
        self.assertEqual(result['image'], ['img/photo.png'])
        self.assertEqual(result['save_name'], 'photo.png')
        self.assertEqual(result['query_id'], '1')

    def test_invalid_vlm_output_preserves_previous_output(self):
        self.output.write_text('previous valid output')
        cases = [dict(self.record, Image_Editing_Prompt='  '),
                 dict(self.record, Image_Editing_Prompt=None),
                 dict(self.record, parse_error='invalid JSON'),
                 dict(self.record, model_output_raw='ERROR: CUDA out of memory')]
        for record in cases:
            with self.subTest(record=record):
                self.write([dict(self.record, index=9), record])
                with self.assertRaises(ValueError):
                    convert(self.source, self.output)
                self.assertEqual(self.output.read_text(), 'previous valid output')

    def test_empty_results_rejected(self):
        self.write([])
        with self.assertRaises(ValueError):
            convert(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_duplicate_ids_or_outputs_rejected(self):
        for second in (dict(self.record), dict(self.record, index=2)):
            self.write([self.record, second])
            with self.assertRaises(ValueError):
                convert(self.source, self.output)

    def test_output_path_escape_rejected(self):
        for name in ('../../overwrite.png', '/tmp/overwrite.png', '..'):
            self.write([dict(self.record, save_name=name)])
            with self.assertRaises(ValueError):
                convert(self.source, self.output)


if __name__ == '__main__':
    unittest.main()
