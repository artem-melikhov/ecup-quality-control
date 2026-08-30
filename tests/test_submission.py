"""Проверки структуры отправки и исключения локальных файлов."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

script = Path(__file__).resolve().parents[1] / 'scripts/build_submission.py'
spec = importlib.util.spec_from_file_location('submission_builder', script)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'project'
        files = ['run.py', 'configs/ensemble.json', 'data/raw/train_data.csv',
                 'data/processed/train_ocr.csv', 'data/processed/train_vision.npz',
                 'data/processed/lv_groups.json', 'data/processed/fewshot_examples.json',
                 'reports/preparation.json', 'models/clip/config.json',
                 'models/clip/preprocessor_config.json', 'models/clip/pytorch_model.bin']
        for name in files:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        for name in ['__init__.py', '__main__.py', 'pipeline.py', 'paths.py', 'ocr.py', 'vision.py', 'prompts.py']:
            path = self.root / 'src' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('')
        (self.root / 'configs/submission.json').write_text(json.dumps({
            'image': 'example/ecup:1.0', 'entry_point': 'python -u run.py'}))
        (self.root / 'configs/local.json').write_text('{"private": "local path"}')

    def test_archive_has_entrypoint_and_project_layout_without_local_files(self):
        path = builder.build_submission(self.root, self.root / 'dist/submission.zip')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            self.assertIn('run.py', names)
            self.assertIn('metadata.json', names)
            self.assertIn('src/pipeline.py', names)
            self.assertIn('data/processed/train_vision.npz', names)
            self.assertIn('models/clip/pytorch_model.bin', names)
            self.assertNotIn('configs/local.json', names)
            self.assertNotIn('configs/submission.json', names)
            self.assertEqual(json.loads(archive.read('metadata.json'))['entry_point'], 'python -u run.py')

    def test_missing_data_does_not_replace_existing_archive(self):
        output = self.root / 'dist/submission.zip'
        output.parent.mkdir()
        output.write_bytes(b'previous archive')
        (self.root / 'data/processed/train_ocr.csv').unlink()
        with self.assertRaises(FileNotFoundError):
            builder.build_submission(self.root, output)
        self.assertEqual(output.read_bytes(), b'previous archive')

    def test_files_outside_project_are_not_packaged(self):
        outside = Path(self.temp.name) / 'outside.json'
        outside.write_text('private')
        config = self.root / 'configs/ensemble.json'
        config.unlink()
        config.symlink_to(outside)
        with self.assertRaises(FileNotFoundError):
            builder.build_submission(self.root, self.root / 'dist/submission.zip')


if __name__ == '__main__':
    unittest.main()
