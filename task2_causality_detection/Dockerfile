FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-tira.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-tira.txt

COPY . ./task2_causality_detection

CMD ["python", "-m", "task2_causality_detection.tira_run"]
