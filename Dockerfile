FROM python:3.13-slim

WORKDIR /app

# install build tools needed for numpy and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# .dockerignore excludes .env, .venv, chromadb, tests - in dockerignore
COPY . .

RUN mkdir -p /tmp/underwriting_uploads

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]nano