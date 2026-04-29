FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    FRONTEND_HOST=0.0.0.0 \
    FRONTEND_PORT=5173 \
    EXTERNAL_PORT=5173

WORKDIR /app

# Build and runtime deps for Python ML stack + Node frontend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    cmake \
    curl \
    git \
    libffi-dev \
    npm \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md /app/
COPY api /app/api
COPY caching /app/caching
COPY commands /app/commands
COPY embeddings /app/embeddings
COPY evaluation /app/evaluation
COPY experiments /app/experiments
COPY generation /app/generation
COPY ingestion /app/ingestion
COPY parser /app/parser
COPY reranking /app/reranking
COPY retrieval /app/retrieval
COPY utils /app/utils
COPY FE /app/FE
COPY main.py /app/main.py
COPY cli.defaults.json /app/cli.defaults.json

RUN poetry install --no-root
RUN npm --prefix /app/FE install

COPY docker/start.sh /app/docker/start.sh
RUN chmod +x /app/docker/start.sh

EXPOSE 5173 8000

CMD ["bash", "/app/docker/start.sh"]
