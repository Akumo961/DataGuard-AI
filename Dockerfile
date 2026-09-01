FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 dataguard
WORKDIR /app

COPY pyproject.toml README.md ./
COPY dataguard ./dataguard
COPY compliance ./compliance
COPY docs ./docs
COPY alembic.ini ./alembic.ini
COPY alembic ./alembic

RUN python -m pip install --upgrade pip \
    && pip install '.[ocr,ner]' \
    && pip download --no-deps --dest /tmp 'https://github.com/explosion/spacy-models/releases/download/xx_ent_wiki_sm-3.8.0/xx_ent_wiki_sm-3.8.0-py3-none-any.whl' \
    && echo '6f3c4b853852ea9e9d2dc76cc950dddb10a7e4c42d813308caefe6c5e8be2f0a  /tmp/xx_ent_wiki_sm-3.8.0-py3-none-any.whl' | sha256sum -c - \
    && pip install /tmp/xx_ent_wiki_sm-3.8.0-py3-none-any.whl \
    && rm -f /tmp/xx_ent_wiki_sm-3.8.0-py3-none-any.whl \
    && chown -R dataguard:dataguard /app

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["sh", "-c", "alembic upgrade head && exec uvicorn dataguard.main:app --host 0.0.0.0 --port 8000"]
