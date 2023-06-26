FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN apt update && \
    apt install -y libpq-dev gcc && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

COPY . .

CMD uvicorn app.api.api:app --host $OPTIMUS_HOST --port $OPTIMUS_PORT --reload