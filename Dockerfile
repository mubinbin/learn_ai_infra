ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY serve.py .

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD [ "python", "-m", "uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000" ]