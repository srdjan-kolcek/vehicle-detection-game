# Placeholder so CI can build the pipeline image. Phase 2 replaces this with the real one.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --create-home appuser
USER appuser

CMD ["python", "-c", "print('pipeline image placeholder')"]
