"""Проверки визуального канала без загрузки больших весов в unit-тестах."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ecup_quality import vision
from ecup_quality import pipeline as run


def vector(index):
    result = np.zeros(vision.DIMENSION, dtype=np.float32)
    result[index] = 1
    return result


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def encoder(self, batch_size=2):
        with patch('ecup_quality.vision.model_fingerprint', return_value='test-signature'):
            return vision.CLIPEncoder(self.root / 'model', self.root / 'cache', 'cpu', batch_size)

    def image(self, item_id, filename, color):
        folder = self.root / 'images' / str(item_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        Image.new('RGB', (80, 40), color).save(path)
        return path

    def test_pooling_cache_and_missing_photos(self):
        self.image(1, '0.jpg', 'white')
        self.image(1, '1.jpg', 'black')
        encoder = self.encoder()
        with patch.object(encoder, '_encode_batch', return_value=np.stack([vector(0), vector(1)])) as encode:
            features, valid = encoder.extract([1, 999], self.root / 'images', log=lambda _: None)
            self.assertEqual(encode.call_count, 1)
        np.testing.assert_allclose(features[0], vision.normalize_vector(vector(0) + vector(1)))
        self.assertEqual(valid.tolist(), [True, False])
        with patch.object(encoder, '_encode_batch', side_effect=AssertionError('Кэш не использован')):
            repeated, mask = encoder.extract([1], self.root / 'images', log=lambda _: None)
        np.testing.assert_allclose(features[0], repeated[0])
        self.assertEqual(mask.tolist(), [True])

    def test_broken_images_and_duplicate_files(self):
        photo = self.image(1, '0.jpg', 'white')
        (photo.parent / '1.jpg').write_bytes(photo.read_bytes())
        (photo.parent / '2.jpg').write_bytes(b'broken')
        encoder = self.encoder()
        with patch.object(encoder, '_encode_batch', return_value=np.stack([vector(0)])) as encode:
            features, valid = encoder.extract([1], self.root / 'images', log=lambda _: None)
            self.assertEqual(len(encode.call_args.args[0]), 1)
        self.assertTrue(valid[0])
        np.testing.assert_array_equal(features[0], vector(0))
        self.assertEqual(vision.product_images(self.root / 'images', '../1'), [])

    def test_precomputed_features_align_ids_and_validate_model(self):
        path = self.root / 'train.npz'
        features = np.stack([vector(0), vector(1)])
        vision.save_features(path, [100, 200], features, [True, False], 'model-A')
        aligned, valid = vision.load_features(path, [200, 300, 100], 'model-A')
        self.assertEqual(valid.tolist(), [False, False, True])
        np.testing.assert_array_equal(aligned[2], vector(0))
        with self.assertRaises(ValueError):
            vision.load_features(path, [100], 'model-B')

    def test_fingerprint_changes_with_weights(self):
        root = self.root / 'model'
        root.mkdir()
        for name in ['config.json', 'preprocessor_config.json']:
            (root / name).write_text('{}')
        (root / 'pytorch_model.bin').write_bytes(b'first model')
        first = vision.model_fingerprint(root)
        (root / 'pytorch_model.bin').write_bytes(b'second model')
        self.assertNotEqual(first, vision.model_fingerprint(root))

    def test_visual_classifier_learns_from_training_labels(self):
        train = pd.DataFrame({'category': ['БАД'] * 8, 'label': [0] * 4 + [1] * 4})
        features = np.stack([vector(0)] * 4 + [vector(1)] * 4)
        test = pd.DataFrame({'category': ['БАД'] * 3})
        probs = vision.fit_visual_classifiers(train, test, features, np.ones(8, bool),
                                             np.stack([vector(0), vector(1), vector(0)]),
                                             np.array([True, True, False]), log=lambda _: None)
        self.assertLess(probs[0], probs[1])
        self.assertTrue(np.isnan(probs[2]))
        train['label'] = 0
        skipped = vision.fit_visual_classifiers(train, test, features, np.ones(8, bool),
                                               np.stack([vector(0)] * 3), np.ones(3, bool), log=lambda _: None)
        self.assertTrue(np.isnan(skipped).all())

    def test_blend_preserves_rows_without_images(self):
        result = vision.blend_visual(np.array([0.8, 0.3]), np.array([0.2, np.nan]), 0.25)
        np.testing.assert_allclose(result, [0.65, 0.3])
        with self.assertRaises(ValueError):
            vision.blend_visual([0.5], [0.5], 1.1)
        with self.assertRaises(ValueError):
            vision.normalize_vector(np.zeros(512))

    def test_encoder_uses_inference_mode(self):
        import torch
        encoder = self.encoder()
        class FakeProcessor:
            def __call__(self, **kwargs):
                return {'pixel_values': torch.zeros(len(kwargs['images']), 3, 224, 224)}
        class FakeModel:
            def __call__(self, **kwargs):
                self.assertions()
                result = type('Output', (), {})()
                result.image_embeds = torch.tensor(np.stack([vector(0)] * len(kwargs['pixel_values'])))
                return result
            def assertions(self):
                assert not torch.is_grad_enabled()
        encoder.torch, encoder.device = torch, 'cpu'
        encoder.model, encoder.processor = FakeModel(), FakeProcessor()
        result = encoder._encode_batch([Image.new('RGB', (10, 10))])
        np.testing.assert_array_equal(result[0], vector(0))

    def test_attach_vision_uses_precomputed_training_and_releases_encoder(self):
        import argparse
        self.image(10, '0.jpg', 'white')
        train = pd.DataFrame({'id': list(range(8)), 'category': ['БАД'] * 8,
                              'label': [0] * 4 + [1] * 4})
        test = pd.DataFrame({'id': [10], 'category': ['БАД']})
        path = self.root / 'train.npz'
        vision.save_features(path, train['id'], np.stack([vector(0)] * 4 + [vector(1)] * 4),
                             np.ones(8, bool), 'signature')
        args = argparse.Namespace(vision='required', vision_weight=.2,
                                  test_data_path=str(self.root / 'test.csv'), images_path=str(self.root / 'images'),
                                  clip_model_path='local-model', train_vision_path=str(path),
                                  vision_cache_dir=str(self.root / 'cache'), vision_device='cpu',
                                  vision_batch_size=2, vision_time_budget=1)
        with patch('ecup_quality.pipeline.load_local_config', return_value={}), patch('ecup_quality.pipeline.CLIPEncoder') as factory:
            encoder = factory.return_value
            encoder.signature = 'signature'
            encoder.extract.return_value = (np.stack([vector(1)]), np.array([True]))
            probabilities = run.attach_vision(train, test, args, lambda _: None)
            self.assertTrue(np.isfinite(probabilities).all())
            self.assertEqual(encoder.extract.call_count, 1)
            encoder.close.assert_called_once()

    def test_missing_model_falls_back_only_in_auto_mode(self):
        import argparse
        self.image(1, '0.jpg', 'white')
        args = argparse.Namespace(vision='auto', vision_weight=.2,
                                  test_data_path=str(self.root / 'test.csv'), images_path=str(self.root / 'images'),
                                  clip_model_path='/does/not/exist', vision_cache_dir=str(self.root / 'cache'),
                                  vision_device='cpu', vision_batch_size=2)
        train = pd.DataFrame({'id': [1], 'label': [0], 'category': ['БАД']})
        test = train.drop(columns='label')
        with patch('ecup_quality.pipeline.load_local_config', return_value={}):
            self.assertTrue(np.isnan(run.attach_vision(train, test, args, lambda _: None)).all())
            args.vision = 'required'
            with self.assertRaises(ValueError):
                run.attach_vision(train, test, args, lambda _: None)

    def test_visual_prediction_affects_submission_and_survives_duplicate_override(self):
        train = pd.DataFrame({'id': [1, 2, 3, 4], 'name': ['a', 'b', 'c', 'd'],
                              'description': ['описание'] * 4, 'category': ['БАД'] * 4,
                              'label': [1, 0, 1, 0]})
        test = train.drop(columns='label').copy()
        output = self.root / 'result.csv'
        args = ['run.py', '-i', 'test.csv', '-o', str(output), '--ocr', 'off',
                '--vision', 'required', '--vision-weight', '1']
        with patch.object(sys, 'argv', args), patch('ecup_quality.pipeline.pd.read_csv', side_effect=[train, test]), \
                patch('ecup_quality.pipeline.tfidf_blocks_nb', return_value=np.array([.9, .8, .2, .1])), \
                patch('ecup_quality.pipeline.attach_vision', return_value=np.array([0., .8, .9, 1.])):
            run.main()
        result = pd.read_csv(output)
        self.assertTrue(result.loc[0, 'result'].endswith('<вердикт>бан'))
        self.assertIn('Учтены визуальные признаки', result.loc[0, 'result'])
        for response in result['result']:
            comment, verdict = response.removeprefix('<комментарий>').split('<вердикт>')
            self.assertTrue(50 <= len(comment) <= 300)
            self.assertIn(verdict, ['бан', 'не бан'])


if __name__ == '__main__':
    unittest.main()
