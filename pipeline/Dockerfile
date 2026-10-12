# Offline video pipeline (doc 04). Never runs at game runtime; started on demand:
#   docker compose run --rm pipeline python -m pipeline <command>
# CPU PyTorch by default (doc 07). requirements.txt pins the CUDA build used for training on the
# host, so torch and torchvision are installed from the CPU index first and skipped from the file.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/ultralytics
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
# ultralytics pulls in the desktop opencv-python, which needs X11 libraries; only the headless build is kept
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -v -E '^(--extra-index-url|torch==|torchvision==)' /tmp/requirements.txt > /tmp/requirements-rest.txt \
    && pip install --no-cache-dir -r /tmp/requirements-rest.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless==5.0.0.93

# data/, weights/ and media/ are mounted at run time; media is the volume the backend serves from
RUN useradd --create-home appuser && mkdir -p /app/media /app/data /app/weights /tmp/ultralytics \
    && chown appuser /app/media /app/data /tmp/ultralytics
COPY . ./pipeline
USER appuser

CMD ["python", "-m", "pipeline", "--help"]
