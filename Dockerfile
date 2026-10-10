# CPU inference image. Build AFTER running `make export` (needs ./artifacts).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/artifacts/model_int8.pt \
    TOKENIZER_PATH=/app/artifacts/tokenizer.json \
    MAX_CONCURRENCY=1

WORKDIR /app

# CPU-only torch wheel (much smaller than the CUDA one). Then the remaining runtime deps.
COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY tinygpt ./tinygpt
COPY serve ./serve
COPY artifacts ./artifacts

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status==200 else 1)"

CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
