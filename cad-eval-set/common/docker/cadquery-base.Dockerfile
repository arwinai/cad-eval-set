# Shared base image for all CadQuery Harbor tasks.
# Build once from the repo root:
#   docker build -f common/docker/cadquery-base.Dockerfile -t cadquery-base:latest .
# Each task's environment/Dockerfile then does `FROM cadquery-base:latest`.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libgomp1 \
        fontconfig \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY env_requirements.txt /tmp/env_requirements.txt
RUN pip install --no-cache-dir -r /tmp/env_requirements.txt

# Shared grading library (common/geom.py, common/harness_base.py, ...).
# Most harnesses do `from common import harness_base`; a few do
# `from harness_base import finalize` directly -- PYTHONPATH covers both.
COPY common /opt/common
ENV PYTHONPATH=/opt:/opt/common

WORKDIR /app
