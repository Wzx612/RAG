FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# CPU-only torch, installed first from PyTorch's CPU index. The default Linux
# wheel bundles CUDA and adds ~2.5 GB to an image that will never see a GPU.
# The version is the same pin as `[project] dependencies` in pyproject.toml;
# keep the two in sync. This is the one intentional duplicate: the CPU index
# and the PyPI index publish different wheels for the same version.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
# The dev, report and api extras ship too: `make test/lint/typecheck`, the
# automatic report and the HTTP service all run in this same image, and a
# reviewer needs them to work without a second build. If the image ever grows a
# serve-only deployment path, split it into a stage.
RUN pip install --no-cache-dir ".[dev,report,api]"

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.cli", "--help"]
