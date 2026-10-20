# Build from the REPO ROOT:  docker build -f docker/Dockerfile -t <user>/prodml-api:0.1.0 .

# ---- Stage 1: builder -- installs the package + its dependencies ----------
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install .

# ---- Stage 2: runtime -- only what's needed to RUN the API -----------------
FROM python:3.11-slim AS runtime
ARG PIP_DEFAULT_TIMEOUT=300
ENV PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT}
ENV PIP_RETRIES=10
ENV PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 appuser
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser models/ /app/models/
WORKDIR /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
