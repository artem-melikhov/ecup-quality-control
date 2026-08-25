"""Проверки подключения фотографий. Метрики качества здесь не считаются."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'submit'))
import ocr
import run

TSV = ('level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n'
       '5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t90\tбез\n'
       '5\t1\t1\t1\t1\t2\t10\t0\t10\t10\t25\tгаза\n')


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def engine(self):
        def startup(args, **kwargs):
            text = 'List of available languages in "/missing" (1):\neng\n'
            if '--version' in args:
                text = 'tesseract test\n'
            return subprocess.CompletedProcess(args, 0, text, '')
        with patch('ocr.shutil.which', return_value='/fake/tesseract'), patch('ocr.subprocess.run', side_effect=startup):
            return ocr.ImageOCR(self.root / 'cache', languages='eng', workers=2)

    def image(self, item_id=1, name='0.jpg', color='white'):
        folder = self.root / 'images' / str(item_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        Image.new('RGB', (80, 40), color).save(path)
        return path

    def test_tsv_preserves_negation_and_deduplicates(self):
        self.assertEqual(ocr._parse_tsv(TSV), 'без газа')
        self.assertEqual(ocr.merge_texts(['Без газа\nБАД', 'без газа\nБАД']), 'Без газа\nБАД')
        with self.assertRaises(ValueError):
            ocr._parse_tsv('plain text instead of TSV')

    def test_cache_uses_content_and_does_not_repeat_ocr(self):
        path = self.image()
        engine = self.engine()
        result = subprocess.CompletedProcess([], 0, TSV.encode(), b'')
        with patch('ocr.subprocess.run', return_value=result) as recognize:
            self.assertEqual(engine.read_image(path, time.monotonic() + 10), ('без газа', 'recognized'))
            self.assertEqual(engine.read_image(path, time.monotonic() + 10), ('без газа', 'cached'))
            self.assertEqual(recognize.call_count, 1)
            Image.new('RGB', (80, 40), 'black').save(path)
            self.assertEqual(engine.read_image(path, time.monotonic() + 10)[1], 'recognized')
            self.assertEqual(recognize.call_count, 2)

    def test_missing_broken_timeout_and_deadline(self):
        engine = self.engine()
        path = self.image()
        broken = self.image(2)
        broken.write_bytes(b'not an image')
        with patch('ocr.subprocess.run', side_effect=subprocess.TimeoutExpired('tesseract', 1)):
            self.assertEqual(engine.read_image(path, time.monotonic() + 10)[1], 'timeout')
        self.assertEqual(engine.read_image(broken, time.monotonic() + 10)[1], 'error')
        self.assertEqual(engine.read_image(path, time.monotonic() - 1)[1], 'deadline')
        self.assertEqual(engine.collect([999], self.root / 'images', log=lambda _: None), [''])
        self.assertEqual(engine.image_paths(self.root / 'images', '../1'), [])
        self.assertEqual(len(engine.image_paths(self.root / 'images', 1.0)), 1)

    def test_exif_orientation_and_resize(self):
        folder = self.root / 'images' / '1'
        folder.mkdir(parents=True)
        exif = Image.Exif()
        exif[274] = 6
        path = folder / '0.jpg'
        Image.new('RGB', (2000, 1000), 'white').save(path, exif=exif)
        engine = self.engine()
        def recognize(args, **kwargs):
            import io
            with Image.open(io.BytesIO(kwargs['input'])) as prepared:
                self.assertEqual(prepared.size, (800, 1600))
            return subprocess.CompletedProcess(args, 0, TSV.encode(), b'')
        with patch('ocr.subprocess.run', side_effect=recognize):
            self.assertEqual(engine.read_image(path, time.monotonic() + 10)[1], 'recognized')

    def test_paths_and_no_symlink_escape(self):
        root = self.root / 'images'
        self.image()
        self.assertEqual(ocr.find_images_root(self.root / 'test.csv'), root.resolve())
        with self.assertRaises(ValueError):
            ocr.find_images_root(self.root / 'test.csv', self.root / 'not-there')
        external = self.root / 'other'
        external.mkdir()
        (root / '2').symlink_to(external, target_is_directory=True)
        self.assertEqual(self.engine().image_paths(root, 2), [])

    def test_ocr_reaches_models_prompts_and_comments(self):
        frame = pd.DataFrame({'id': [1], 'category': ['БАД'], 'name': ['Товар'],
                              'description': ['<p>Описание</p>'], 'ocr_text': ['БАД витамины']})
        run.prepare_features(frame, {'БАД'})
        self.assertIn('бад витамины', frame.loc[0, 'f_txt'])
        self.assertTrue(frame.loc[0, 'ocr_used'])
        self.assertEqual(frame.loc[0, 'desc_raw'], '<p>Описание</p>')
        frame.loc[0, 'desc'] = 'Описание ' * 1000
        self.assertIn('БАД витамины', run.prompt_description(frame.loc[0]))
        response = run.build_submission(frame, pd.Series([1])).loc[0, 'result']
        self.assertIn('Распознано на фото (OCR)', response)
        comment, verdict = response.removeprefix('<комментарий>').split('<вердикт>')
        self.assertTrue(50 <= len(comment) <= 300)
        self.assertEqual(verdict, 'не бан')
        # Без обучающих фото текст линейной модели остаётся прежним.
        run.prepare_features(frame, ())
        self.assertNotIn('бад витамины', frame.loc[0, 'f_txt'])
        self.assertNotIn('Распознано на фото', run.build_submission(frame, pd.Series([1])).loc[0, 'result'])
        self.assertIn('БАД витамины', run.prompt_description(frame.loc[0]))

    def test_linear_models_receive_identical_feature_layout(self):
        train = pd.DataFrame({'name': ['Товар'] * 8, 'description': ['описание'] * 8,
                              'category': ['БАД'] * 8,
                              'ocr_text': ['витамины бад'] * 4 + ['протеин спорт'] * 4})
        test = train.iloc[[0, 4]].copy()
        run.prepare_features(train, {'БАД'})
        run.prepare_features(test, {'БАД'})
        labels = np.array([1] * 4 + [0] * 4)
        probabilities = run.tfidf_blocks_nb(train, test, labels, run.CFG1)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_precomputed_training_text_is_mapped_by_id(self):
        self.image()
        training_csv = self.root / 'train_ocr.csv'
        pd.DataFrame({'id': [2], 'ocr_text': ['Витамины на упаковке']}).to_csv(training_csv, index=False)
        train = pd.DataFrame({'id': [1, 2], 'category': ['БАД', 'БАД']})
        test = pd.DataFrame({'id': [3], 'category': ['БАД']})
        args = argparse.Namespace(ocr='required', test_data_path=str(self.root / 'test.csv'),
                                  images_path=str(self.root / 'images'), train_images_path=None,
                                  train_ocr_path=str(training_csv), ocr_cache_dir=str(self.root / 'cache'),
                                  tessdata_dir=None, ocr_languages='rus+eng', ocr_workers=1, ocr_time_budget=1)
        with patch('run.load_local_config', return_value={}), patch('run.ImageOCR') as engine:
            engine.return_value.collect.return_value = ['БАД на фото теста']
            run.attach_ocr(train, test, args, lambda _: None)
            self.assertEqual(train['ocr_text'].tolist(), ['', 'Витамины на упаковке'])
            self.assertEqual(test['ocr_text'].tolist(), ['БАД на фото теста'])
            self.assertEqual(engine.return_value.collect.call_count, 1)

    def test_optional_backend_failure_and_required_mode(self):
        self.image()
        train = pd.DataFrame({'id': [1], 'category': ['БАД']})
        test = train.copy()
        args = argparse.Namespace(ocr='auto', test_data_path=str(self.root / 'test.csv'),
                                  images_path=str(self.root / 'images'), train_images_path=None, train_ocr_path=None,
                                  ocr_cache_dir=str(self.root / 'cache'), tessdata_dir=None,
                                  ocr_languages='rus+eng', ocr_workers=1, ocr_time_budget=1)
        with patch('run.load_local_config', return_value={}), patch('run.ImageOCR', side_effect=RuntimeError('missing')):
            run.attach_ocr(train, test, args, lambda _: None)
            self.assertEqual(test.loc[0, 'ocr_text'], '')
            args.ocr = 'required'
            with self.assertRaises(RuntimeError):
                run.attach_ocr(train, test, args, lambda _: None)


if __name__ == '__main__':
    unittest.main()
