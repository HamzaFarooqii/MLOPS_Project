FROM python:3.10-slim
WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10

COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=${PIP_DEFAULT_TIMEOUT} --retries=${PIP_RETRIES} -r /app/requirements.txt

COPY server/app.py /app/app.py
COPY model_download/ /app/model_download/

ENV PORT=8000
ENV MODEL_LOCAL_PATH=/app/model_download
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
