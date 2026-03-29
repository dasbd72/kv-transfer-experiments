FROM python:3.12

# Install uv for speed
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
&& $HOME/.local/bin/uv venv /opt/venv --python ${PYTHON_VERSION} \
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
    uv pip install transformers huggingface_hub
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu130

# Copy the rest of the application code
WORKDIR /app

COPY header.py kv_layout.py memfd_transfer.py socket_transfer.py cuda_ipc_transfer.py ./
