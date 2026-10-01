# Shared base image for all Blender Harbor tasks.
# Build once from the repo root:
#   docker build -f common/docker/blender-base.Dockerfile -t blender-base:latest .
# Each task's environment/Dockerfile then does `FROM blender-base:latest`.
#
# Blender ships no official Docker image, so this installs the official
# Linux build at build time and puts `blender` on PATH; tasks and harnesses
# run it headless (`blender -b file.blend --python script.py`).
#
# THE VERSION IS PINNED ON PURPOSE. 5.2.2 LTS is what task 100's baseline
# was frozen with, and its P7 compares modifier settings, which Blender
# renames between versions: under 4.5 the reference loses P7 because the
# Boolean solver's options are named differently. Linux 5.2.2 gives the
# same subscores as Windows 5.2.2 on every file of that task.
#
# download.blender.org refuses scripted downloads (HTTP 403), so the build
# is fetched from Blender's official mirrors and checked against the
# checksum Blender publishes (blender-5.2.2.sha256).

FROM python:3.11-slim

# The only system libraries the Blender binary links against that a slim
# image lacks (read off `ldd blender`); X and GL are needed even headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates xz-utils \
        libgl1 libx11-6 libxext6 libxfixes3 libxi6 libxrender1 \
        libxkbcommon0 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/*

ARG BLENDER_VERSION=5.2.2
ARG BLENDER_SHA256=84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168
RUN set -e; \
    f="blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    for m in https://mirrors.ocf.berkeley.edu/blender \
             https://ftp.nluug.nl/pub/graphics/blender \
             https://mirror.clarkson.edu/blender; do \
        wget -q "$m/release/Blender${BLENDER_VERSION%.*}/$f" -O "/tmp/$f" && break; \
    done; \
    echo "${BLENDER_SHA256}  /tmp/$f" | sha256sum -c -; \
    tar -xJf "/tmp/$f" -C /opt; \
    mv "/opt/blender-${BLENDER_VERSION}-linux-x64" /opt/blender; \
    rm "/tmp/$f"; \
    ln -s /opt/blender/blender /usr/local/bin/blender; \
    blender -b --version | head -1

# The harnesses find Blender through this variable first.
ENV BLENDER=/usr/local/bin/blender

# Grading runs in this Python, outside Blender, and needs numpy and scipy.
# Their pins are read from env_requirements.txt so the two cannot drift;
# the rest of that file (CadQuery, cq_gears built from git) is not needed
# here and would need a compiler.
COPY env_requirements.txt /tmp/env_requirements.txt
RUN grep -E '^(numpy|scipy)==' /tmp/env_requirements.txt > /tmp/blender_requirements.txt \
    && pip install --no-cache-dir -r /tmp/blender_requirements.txt

# Shared grading library (common/harness_base.py, ...).
COPY common /opt/common
ENV PYTHONPATH=/opt:/opt/common

WORKDIR /app
