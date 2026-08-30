"""Собрать конкурсный ZIP из структуры репозитория."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_submission(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    files = {
        'run.py': root / 'run.py',
        'metadata.json': root / 'configs/submission.json',
        'configs/ensemble.json': root / 'configs/ensemble.json',
        'data/raw/train_data.csv': root / 'data/raw/train_data.csv',
        'data/processed/train_ocr.csv': root / 'data/processed/train_ocr.csv',
        'data/processed/train_vision.npz': root / 'data/processed/train_vision.npz',
        'data/processed/lv_groups.json': root / 'data/processed/lv_groups.json',
        'data/processed/fewshot_examples.json': root / 'data/processed/fewshot_examples.json',
        'reports/preparation.json': root / 'reports/preparation.json',
        'models/clip/config.json': root / 'models/clip/config.json',
        'models/clip/preprocessor_config.json': root / 'models/clip/preprocessor_config.json',
    }
    package = root / 'src'
    for source in package.rglob('*.py'):
        files[str(source.relative_to(root))] = source
    for name in ['__init__.py', '__main__.py', 'pipeline.py', 'paths.py', 'ocr.py', 'vision.py', 'prompts.py']:
        if not (package / name).is_file():
            raise FileNotFoundError(f'Не найден модуль: {package / name}')
    model_root = root / 'models/clip'
    # Если есть оба формата, Transformers выбирает safetensors.
    weights = model_root / 'model.safetensors'
    if not weights.is_file():
        weights = model_root / 'pytorch_model.bin'
    files[str(weights.relative_to(root))] = weights
    for name in ['README.md', 'LICENSE', 'LICENSE.txt']:
        if (model_root / name).is_file():
            files['models/clip/' + name] = model_root / name
    for name, source in files.items():
        if not source.is_file() or not source.resolve().is_relative_to(root):
            raise FileNotFoundError(f'Нет файла для отправки: {source}')
    metadata = json.loads(files['metadata.json'].read_text(encoding='utf-8'))
    if not isinstance(metadata.get('image'), str) or not metadata['image'].strip():
        raise ValueError('В configs/submission.json не указан Docker-образ')
    if metadata.get('entry_point') != 'python -u run.py':
        raise ValueError('Ожидается точка входа python -u run.py')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.zip', delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for name, source in sorted(files.items()):
                method = zipfile.ZIP_STORED if source.suffix in ('.bin', '.safetensors', '.npz') else zipfile.ZIP_DEFLATED
                archive.write(source, arcname=name, compress_type=method)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise ValueError('Не удалось проверить целостность ZIP')
        if temporary.stat().st_size > 5_000_000_000:
            raise ValueError('Архив превышает лимит 5 GB')
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description='Собрать архив для проверки решения')
    parser.add_argument('--output', default=str(PROJECT_ROOT / 'dist/submission.zip'))
    args = parser.parse_args()
    destination = build_submission(PROJECT_ROOT, args.output)
    print(f'Архив: {destination} ({destination.stat().st_size / 1024**2:.1f} MB)')


if __name__ == '__main__':
    main()
