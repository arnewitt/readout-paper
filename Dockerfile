# syntax=docker/dockerfile:1

# Both stages share one base image so the virtualenv built in the builder keeps
# pointing at an interpreter that still exists at runtime.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.7 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first, in their own layer: they are the slow, ~1 GB part, and
# they only need to be rebuilt when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Then the project itself. Installed non-editable so nothing in the runtime
# image depends on /app/src still being present.
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable


FROM ${PYTHON_IMAGE} AS runtime

# espeak-ng ships inside the espeakng-loader wheel, so the runtime image needs
# no apt packages at all beyond what python:slim already has.
# The HF cache directory must exist and be owned by `app` in the image: a fresh
# named volume mounted there inherits the ownership of the underlying path, and
# a missing path yields a root-owned volume the app user cannot write to.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /home/app/.cache/huggingface \
    && chown -R app:app /data /home/app/.cache

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

USER app

# Model weights land in HF_HOME on first run (~350 MB). Mount a volume there or
# every fresh container downloads them again.
VOLUME ["/home/app/.cache/huggingface"]

# WAVs written with a relative path land here.
WORKDIR /data

EXPOSE 8765

ENTRYPOINT ["readout-paper"]
CMD ["--ui", "--host", "0.0.0.0"]
