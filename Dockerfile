# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

CMD ["python", "script_runner_bot_v8.py"]
