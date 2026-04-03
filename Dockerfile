FROM nvidia/cuda:13.2.0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
python3 python3-pip python3-dev curl build-essential \
&& rm -rf /var/lib/apt/lists/*

# Install uv for speed
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
&& $HOME/.local/bin/uv venv /opt/venv --python 3.12 \
&& rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
&& ln -s /opt/venv/bin/python3 /usr/bin/python3 \
&& ln -s /opt/venv/bin/python3-config /usr/bin/python3-config \
&& ln -s /opt/venv/bin/pip /usr/bin/pip \
&& python3 --version && python3 -m pip --version

# Activate virtual environment and add uv to PATH
ENV PATH="/opt/venv/bin:/root/.local/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Environment for uv
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install transformers huggingface_hub ninja
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu130

ARG torch_cuda_arch_list='7.5 8.0 8.9 9.0 10.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# Copy the rest of the application code
WORKDIR /app

COPY setup.py pyproject.toml header.py kv_layout.py shm_transfer.py memfd_transfer.py socket_transfer.py cuda_ipc_transfer.py ./
COPY csrc/*.cu csrc/*.cpp ./csrc/
RUN uv pip install setuptools wheel \
    && uv pip install . --no-build-isolation
