# Stage 1: Base image with Python and uv
FROM python:3.11-slim AS base

# Install uv
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Stage 2: Application image
FROM base AS app
WORKDIR /app

# Copy the project files
COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --extra docker

COPY model/model.onnx model/model.onnx
COPY inference/ inference/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uv", "run", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
