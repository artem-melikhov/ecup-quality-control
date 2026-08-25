"""Локальное распознавание надписей на фотографиях товаров через Tesseract."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import warnings

SCHEMA = 2
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def find_images_root(csv_path, explicit=None):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f'Нет папки изображений: {path}')
        return path
    csv_path = Path(csv_path).resolve()
    for path in (csv_path.parent / 'images', csv_path.with_suffix('') / 'images', csv_path.with_suffix('')):
        if path.is_dir():
            return path
    return None


def load_local_config():
    here = Path(__file__).resolve().parent
    for path in (here / 'ocr.local.json', here.parent / 'ocr.local.json'):
        if path.is_file():
            with path.open(encoding='utf-8') as stream:
                return json.load(stream)
    return {}


def merge_texts(texts, max_chars=2000):
    lines, seen = [], set()
    for text in texts:
        for line in text.splitlines():
            line = re.sub(r'\s+', ' ', line).strip()
            key = line.casefold()
            if line and key not in seen:
                lines.append(line)
                seen.add(key)
    return '\n'.join(lines)[:max_chars]


def _parse_tsv(text):
    if not text.startswith("level\t"):
        raise ValueError("Tesseract не вернул TSV")
    lines = defaultdict(list)
    for row in csv.DictReader(io.StringIO(text), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row.get('level') != '5' or not row.get('text', '').strip():
            continue
        try:
            confidence = float(row['conf'])
        except (ValueError, KeyError):
            continue
        key = tuple(row.get(field) for field in ('page_num', 'block_num', 'par_num', 'line_num'))
        lines[key].append((row['text'].strip(), confidence))
    result = []
    for words in lines.values():
        # Отбрасываем целую строку, а не отдельные слова: так не теряется «не» или «без».
        line = ' '.join(word for word, _ in words)
        if sum(conf for _, conf in words) / len(words) >= 35 and sum(c.isalpha() for c in line) >= 2:
            result.append(line)
    return merge_texts(['\n'.join(result)])


class ImageOCR:
    def __init__(self, cache_dir, tessdata_dir=None, languages='rus+eng', workers=4,
                 max_images=5, max_chars=2000, timeout=8, max_side=1600):
        from PIL import Image, ImageOps
        self.Image, self.ImageOps = Image, ImageOps
        self.binary = shutil.which('tesseract')
        if not self.binary:
            raise RuntimeError('Tesseract не установлен')
        self.tessdata_dir = str(Path(tessdata_dir).resolve()) if tessdata_dir else None
        self.languages = languages
        self.workers = max(1, min(int(workers), 8))
        self.max_images, self.max_chars = max_images, max_chars
        self.timeout, self.max_side = timeout, max_side
        if min(max_images, max_chars, timeout, max_side) <= 0:
            raise ValueError('Лимиты OCR должны быть больше нуля')
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        args = [self.binary, '--list-langs'] + self._data_args()
        available = subprocess.run(args, capture_output=True, text=True, timeout=10, check=True).stdout
        missing = set(languages.split('+')) - set(available.splitlines())
        if missing:
            raise RuntimeError('Нет языков Tesseract: ' + ', '.join(sorted(missing)))
        version = subprocess.run([self.binary, '--version'], capture_output=True, text=True,
                                 timeout=10, check=True).stdout.splitlines()[0]
        # Учитываем сами языковые веса, чтобы замена модели не оставила старый кэш.
        data_path = self.tessdata_dir
        if not data_path:
            match = re.search(r'"([^"]+)"', available.splitlines()[0])
            data_path = match.group(1) if match else ''
        model_hashes = []
        for lang in languages.split('+'):
            model = Path(data_path) / (lang + '.traineddata')
            model_hashes.append(hashlib.sha256(model.read_bytes()).hexdigest() if model.is_file() else lang)
        self.signature = json.dumps([SCHEMA, version, languages, model_hashes, max_side, 11, 35]).encode()

    def _data_args(self):
        return ['--tessdata-dir', self.tessdata_dir] if self.tessdata_dir else []

    def image_paths(self, root, item_id):
        root = Path(root).resolve()
        item_id = str(item_id)
        if re.fullmatch(r'\d+\.0', item_id):
            item_id = item_id[:-2]
        if not re.fullmatch(r'\d+', item_id):
            return []
        folder = root / item_id
        if not folder.is_dir() or not folder.resolve().is_relative_to(root):
            return []
        files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
                 and p.is_file() and p.resolve().is_relative_to(root)]
        return sorted(files, key=lambda p: (len(p.stem), p.name))[:self.max_images]

    def read_image(self, path, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return '', 'deadline'
        try:
            if path.stat().st_size > 30 * 1024 * 1024:
                return '', 'oversize'
            data = path.read_bytes()
            key = hashlib.sha256(self.signature + data).hexdigest()
            cached = self.cache_dir / (key + '.json')
            if cached.is_file():
                try:
                    payload = json.loads(cached.read_text(encoding='utf-8'))
                    if isinstance(payload.get('text'), str):
                        return payload['text'], 'cached'
                except (ValueError, OSError, AttributeError):
                    pass
            with warnings.catch_warnings():
                warnings.simplefilter('error', self.Image.DecompressionBombWarning)
                with self.Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > 25_000_000:
                        return '', 'oversize'
                    image = self.ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((self.max_side, self.max_side), self.Image.Resampling.LANCZOS)
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return '', 'deadline'
            env = dict(os.environ, OMP_THREAD_LIMIT='1')
            command = [self.binary, 'stdin', 'stdout', '-l', self.languages,
                       '--oem', '1', '--psm', '11'] + self._data_args() + ['-c', 'tessedit_create_tsv=1']
            result = subprocess.run(command, input=buffer.getvalue(), capture_output=True,
                                    timeout=min(self.timeout, remaining), env=env, check=True)
            text = _parse_tsv(result.stdout.decode('utf-8', errors='replace'))
            # Атомарная запись позволяет использовать один кэш в нескольких запусках.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.cache_dir,
                                                 suffix='.tmp', delete=False) as stream:
                    temporary = Path(stream.name)
                    json.dump({'text': text}, stream, ensure_ascii=False)
                temporary.replace(cached)
            except OSError:
                pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return text, 'recognized'
        except subprocess.TimeoutExpired:
            return '', 'timeout'
        except (OSError, ValueError, subprocess.CalledProcessError, self.Image.DecompressionBombWarning,
                self.Image.DecompressionBombError):
            return '', 'error'

    def collect(self, ids, root, time_budget=60, log=print):
        """Вернуть текст для каждого ID в исходном порядке. Ноль отключает общий дедлайн."""
        if time_budget < 0:
            raise ValueError('Лимит времени OCR не может быть отрицательным')
        root = Path(root).resolve()
        deadline = time.monotonic() + time_budget if time_budget else float('inf')

        def read_card(item_id):
            stats = Counter()
            try:
                paths = self.image_paths(root, item_id)
            except OSError:
                return '', Counter(error=1)
            if not paths:
                return '', Counter(missing=1)
            texts = []
            for path in paths:
                text, status = self.read_image(path, deadline)
                stats[status] += 1
                texts.append(text)
                if status == 'deadline':
                    break
            return merge_texts(texts, self.max_chars), stats

        stats, texts = Counter(), []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for text, status in pool.map(read_card, ids):
                texts.append(text)
                stats.update(status)
        log(f'OCR: текст у {sum(bool(t) for t in texts)}/{len(texts)} карточек, {dict(stats)}')
        return texts


def main():
    parser = argparse.ArgumentParser(description='Подготовить OCR-кэш до запуска моделей')
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--images-path', required=True)
    parser.add_argument('--cache-dir', default=str(Path(__file__).resolve().parent / '.ocr-cache'))
    parser.add_argument('--tessdata-dir')
    parser.add_argument('--languages', default='rus+eng')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--time-budget', type=float, default=0)
    parser.add_argument('--output-path', help='Необязательный CSV со столбцами id и ocr_text')
    args = parser.parse_args()
    config = load_local_config()
    root = find_images_root(args.data_path, args.images_path)
    engine = ImageOCR(args.cache_dir, args.tessdata_dir or config.get('tessdata_dir'),
                      args.languages, args.workers)
    with open(args.data_path, encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [row['id'] for row in rows]
    texts = engine.collect(ids, root, args.time_budget)
    if args.output_path:
        with open(args.output_path, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'ocr_text'])
            writer.writerows(zip(ids, texts))


if __name__ == '__main__':
    main()
