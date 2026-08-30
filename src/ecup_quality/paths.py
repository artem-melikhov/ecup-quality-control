"""Пути проекта одинаковы в репозитории и в архиве отправки."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('ECUP_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
CONFIGS = ROOT / 'configs'
RAW_DATA = ROOT / 'data' / 'raw'
PROCESSED_DATA = ROOT / 'data' / 'processed'
MODELS = ROOT / 'models'
CACHE = ROOT / '.cache'
TRAIN_DATA = RAW_DATA / 'train_data.csv'
TRAIN_OCR = PROCESSED_DATA / 'train_ocr.csv'
TRAIN_VISION = PROCESSED_DATA / 'train_vision.npz'
LOCAL_CONFIG = CONFIGS / 'local.json'
