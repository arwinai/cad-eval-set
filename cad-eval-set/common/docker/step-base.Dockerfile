# Shared base image for all STEP program Harbor tasks.
# Build once from the repo root:
#   docker build -f common/docker/step-base.Dockerfile -t step-base:latest .
# Each task's environment/Dockerfile then does `FROM step-base:latest`.
#
# STEP tasks grade a text report (regex + LLM-judged via common/call_llm.py),
# not CAD geometry -- no CAD toolchain needed here, unlike
# cadquery-base/freecad-base.

FROM python:3.11-slim

COPY env_requirements.txt /tmp/env_requirements.txt
RUN pip install --no-cache-dir -r /tmp/env_requirements.txt

# Shared grading library (common/harness_base.py, common/call_llm.py, ...).
COPY common /opt/common
ENV PYTHONPATH=/opt:/opt/common

WORKDIR /app
