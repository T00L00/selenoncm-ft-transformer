# syntax=docker/dockerfile:1.6

# Official PyTorch CUDA image (devel includes compilers/tooling; good for dev)
# You can bump this tag as needed.
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

# System deps useful for ML/dev workflows
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    cmake \
    ninja-build \
    vim \
    tmux \
    htop \
    tree \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (nice for dev + avoids root-owned files on bind mounts)
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# Python ergonomics
RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
 && python -m pip install --no-cache-dir \
      ipython \
      jupyterlab \
      rich \
      tensorboard

# Workspace
WORKDIR /workspace

# Optional: install your project dependencies (keeps rebuilds fast when code changes)
# If you use requirements.txt:
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# If you use pyproject.toml (setuptools/poetry), you can adapt accordingly.

# Default environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to dev user by default
USER ${USERNAME}

# Keep the container alive for interactive development; override in compose if desired
CMD ["bash", "-lc", "sleep infinity"]
