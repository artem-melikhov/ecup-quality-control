FROM odsai/ecup26-quality-baseline:1.0

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir 'Pillow>=10.1,<13'
ENV OMP_THREAD_LIMIT=1
