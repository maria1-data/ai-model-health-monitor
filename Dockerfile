
FROM python:3.12.10-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir scikit-learn==1.8.0 mlflow==3.18.0

COPY config/ ./config/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Create a Linux-compatible MLflow registry from the existing trained model.
RUN python scripts/register_docker_model.py

EXPOSE 8000

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
