"""Точка входа для локального запуска и конкурсного контейнера."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from ecup_quality.pipeline import main

if __name__ == '__main__':
    main()
