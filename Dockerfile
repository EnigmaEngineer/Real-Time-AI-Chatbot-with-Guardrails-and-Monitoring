FROM python:3.12-slim AS base

RUN groupadd -r chatbot && useradd -r -g chatbot chatbot

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir .[prod]

COPY src/ src/
COPY config.yaml .

RUN mkdir -p data && chown -R chatbot:chatbot /app

USER chatbot

EXPOSE 8000 9090

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
