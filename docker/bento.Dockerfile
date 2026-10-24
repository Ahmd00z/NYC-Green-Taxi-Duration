# BentoML model server image (used twice in the canary compose: v1 and v2).
FROM python:3.12-slim
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY serving ./serving
COPY models ./models
RUN pip install --no-cache-dir ".[serving]"
EXPOSE 3000
CMD ["bentoml", "serve", "serving.service:TaxiDuration", "--port", "3000"]
