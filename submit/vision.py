"""Признаки фотографий из замороженного CLIP и классификация товаров по ним."""
import argparse
import hashlib
from pathlib import Path
import re
import tempfile
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

SCHEMA = 1
DIMENSION = 512
SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def model_fingerprint(model_path):
    """Подпись зависит от весов и обработки изображения, но не от пути на диске."""
    root = Path(model_path)
    files = [root / 'config.json', root / 'preprocessor_config.json']
    weights = root / 'model.safetensors'
    if not weights.is_file():
        weights = root / 'pytorch_model.bin'
    files.append(weights)
    digest = hashlib.sha256(f'clip-vit-b32:{SCHEMA}:mean-l2:exif:5:1600'.encode())
    for path in files:
        if not path.is_file():
            raise ValueError(f'Не хватает файла CLIP: {path}')
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (DIMENSION,) or not np.isfinite(vector).all():
        raise ValueError('Неверный вектор CLIP')
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Нулевой вектор CLIP')
    return vector / norm


def product_images(root, item_id):
    root = Path(root).resolve()
    item_id = str(item_id)
    if re.fullmatch(r'\d+\.0', item_id):
        item_id = item_id[:-2]
    if not re.fullmatch(r'\d+', item_id):
        return []
    folder = root / item_id
    if not folder.is_dir() or not folder.resolve().is_relative_to(root):
        return []
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES
             and p.resolve().is_relative_to(root)]
    return sorted(files, key=lambda p: (len(p.stem), p.name))[:5]


class CLIPEncoder:
    def __init__(self, model_path, cache_dir, device='auto', batch_size=8):
        self.model_path = Path(model_path).expanduser().resolve()
        self.signature = model_fingerprint(self.model_path)
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if batch_size < 1:
            raise ValueError('Размер батча CLIP должен быть больше нуля')
        if device not in ('auto', 'cpu', 'cuda', 'mps'):
            raise ValueError('Неизвестное устройство CLIP')
        self.requested_device, self.batch_size = device, batch_size
        self.model = self.processor = self.torch = None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        device = self.requested_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.torch = torch
        self.processor = CLIPImageProcessor.from_pretrained(str(self.model_path), local_files_only=True)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            str(self.model_path), local_files_only=True, trust_remote_code=False
        ).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device

    def _encode_batch(self, images):
        self._load_model()
        inputs = self.processor(images=images, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            vectors = self.model(**inputs).image_embeds.float().cpu().numpy()
        return np.stack([normalize_vector(vector) for vector in vectors])

    def _save_cache(self, key, vector):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.save(stream, vector, allow_pickle=False)
            temporary.replace(self.cache_dir / (key + '.npy'))
        except OSError:
            pass
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def extract(self, ids, images_root, time_budget=60, log=print):
        """Среднее L2-нормированных признаков фото, затем повторная нормализация."""
        from PIL import Image, ImageOps
        import io
        if not np.isfinite(time_budget) or time_budget < 0:
            raise ValueError('Лимит времени CLIP не может быть отрицательным')
        ids = list(ids)
        deadline = time.monotonic() + time_budget if time_budget else float('inf')
        sums = np.zeros((len(ids), DIMENSION), dtype=np.float32)
        counts = np.zeros(len(ids), dtype=np.int32)
        pending, cached, errors = [], 0, 0

        def flush():
            if not pending:
                return
            try:
                vectors = self._encode_batch([item[2] for item in pending])
                for (row, key, _), vector in zip(pending, vectors):
                    sums[row] += vector
                    counts[row] += 1
                    self._save_cache(key, vector)
            finally:
                for _, _, image in pending:
                    image.close()
                pending.clear()

        try:
            for row, item_id in enumerate(ids):
                if time.monotonic() >= deadline:
                    break
                seen = set()
                try:
                    paths = product_images(images_root, item_id)
                except OSError:
                    errors += 1
                    continue
                for path in paths:
                    if time.monotonic() >= deadline:
                        break
                    try:
                        if path.stat().st_size > 30 * 1024 * 1024:
                            errors += 1
                            continue
                        data = path.read_bytes()
                        key = hashlib.sha256(self.signature.encode() + data).hexdigest()
                        if key in seen:
                            continue
                        seen.add(key)
                        cache = self.cache_dir / (key + '.npy')
                        if cache.is_file():
                            try:
                                vector = normalize_vector(np.load(cache, allow_pickle=False))
                                sums[row] += vector
                                counts[row] += 1
                                cached += 1
                                continue
                            except (ValueError, OSError):
                                pass
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', Image.DecompressionBombWarning)
                            with Image.open(io.BytesIO(data)) as source:
                                if source.width * source.height > 25_000_000:
                                    errors += 1
                                    continue
                                image = ImageOps.exif_transpose(source).convert('RGB')
                            image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                        pending.append((row, key, image))
                    except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
                        errors += 1
                        continue
                    if len(pending) >= self.batch_size:
                        flush()
            # Изображения, подготовленные до дедлайна, обрабатываются последним батчем.
            flush()
        finally:
            for _, _, image in pending:
                image.close()
            pending.clear()
        valid = counts > 0
        for row in np.flatnonzero(valid):
            sums[row] = normalize_vector(sums[row] / counts[row])
        log(f'CLIP: признаки у {int(valid.sum())}/{len(ids)} товаров, фото из кэша {cached}, ошибок {errors}')
        return sums, valid

    def close(self):
        self.model = self.processor = None
        if self.torch is not None and getattr(self, 'device', None) == 'cuda':
            self.torch.cuda.empty_cache()


def save_features(path, ids, features, valid, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, ids=np.asarray(ids, dtype=str), features=features,
                                valid=np.asarray(valid, dtype=bool), signature=np.array(signature), schema=np.array(SCHEMA))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_features(path, ids, signature):
    with np.load(path, allow_pickle=False) as data:
        required = {'schema', 'signature', 'ids', 'features', 'valid'}
        if not required <= set(data.files):
            raise ValueError('В файле признаков CLIP не хватает полей')
        if int(data['schema']) != SCHEMA or str(data['signature']) != signature:
            raise ValueError('Признаки трейна рассчитаны другой моделью или настройками CLIP')
        stored_ids = data['ids'].astype(str)
        features = data['features'].astype(np.float32)
        valid = data['valid']
    if (stored_ids.ndim != 1 or features.shape != (len(stored_ids), DIMENSION)
            or valid.shape != (len(stored_ids),) or valid.dtype != np.bool_
            or len(set(stored_ids)) != len(stored_ids) or not np.isfinite(features).all()):
        raise ValueError('Повреждённый файл признаков CLIP')
    lookup = {key: i for i, key in enumerate(stored_ids)}
    ids = list(ids)
    aligned = np.zeros((len(ids), DIMENSION), dtype=np.float32)
    available = np.zeros(len(ids), dtype=bool)
    for row, item_id in enumerate(ids):
        stored = lookup.get(str(item_id))
        if stored is not None and valid[stored]:
            aligned[row] = normalize_vector(features[stored])
            available[row] = True
    return aligned, available


def fit_visual_classifiers(train, test, train_features, train_valid, test_features, test_valid, log=print):
    """Обучить отдельную логистическую регрессию на фото для каждой категории."""
    probabilities = np.full(len(test), np.nan)
    for category in sorted(set(test['category'])):
        fit_mask = (train['category'].to_numpy() == category) & train_valid
        predict_mask = (test['category'].to_numpy() == category) & test_valid
        if not predict_mask.any():
            continue
        labels = train.loc[fit_mask, 'label'].to_numpy()
        classes, counts = np.unique(labels, return_counts=True)
        if set(classes) != {0, 1} or counts.min() < 2:
            log(f'CLIP {category}: мало обучающих фото, нужны хотя бы два товара каждого класса')
            continue
        model = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear',
                                   max_iter=2000, random_state=42)
        model.fit(train_features[fit_mask], labels)
        probabilities[predict_mask] = model.predict_proba(test_features[predict_mask])[:, 1]
        log(f'CLIP {category}: обучение на {int(fit_mask.sum())}, предсказание для {int(predict_mask.sum())}')
    return probabilities


def blend_visual(base_probabilities, visual_probabilities, weight=0.2):
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Вес визуальной модели должен быть от 0 до 1')
    base = np.asarray(base_probabilities, dtype=float)
    visual = np.asarray(visual_probabilities, dtype=float)
    if base.shape != visual.shape:
        raise ValueError('Размеры предсказаний моделей не совпадают')
    result = base.copy()
    mask = np.isfinite(visual)
    if np.any((visual[mask] < 0) | (visual[mask] > 1)):
        raise ValueError('Неверные вероятности визуальной модели')
    result[mask] = (1 - weight) * base[mask] + weight * visual[mask]
    return result


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description='Подготовить визуальные признаки трейна через CLIP')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--images-path', required=True)
    parser.add_argument('--model-path', required=True, help='Локальная папка CLIP ViT-B/32')
    parser.add_argument('--output-path', default=str(Path(__file__).resolve().parent / 'train_vision.npz'))
    parser.add_argument('--cache-dir', default=str(Path(__file__).resolve().parent / '.vision-cache'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--time-budget', type=float, default=0)
    args = parser.parse_args()
    frame = pd.read_csv(args.data_path)
    if frame['id'].duplicated().any():
        parser.error('ID товаров должны быть уникальны')
    encoder = CLIPEncoder(args.model_path, args.cache_dir, args.device, args.batch_size)
    try:
        features, valid = encoder.extract(frame['id'].tolist(), args.images_path, args.time_budget)
        save_features(args.output_path, frame['id'].tolist(), features, valid, encoder.signature)
    finally:
        encoder.close()


if __name__ == '__main__':
    main()
