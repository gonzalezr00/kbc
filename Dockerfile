FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git curl wget vim build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -s /bin/bash dev
WORKDIR /home/dev/project

COPY requirements/base.txt requirements/base.txt
RUN uv pip install --system --no-cache -r requirements/base.txt

EXPOSE 8888
USER dev
