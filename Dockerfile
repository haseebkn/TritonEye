FROM python:3.12-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies for geospatial libraries and building wheel extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy configuration, agent modules and tests. tests/ is required by the default
# CMD below; without it the image's own smoke test fails on a missing path.
COPY configs/ /app/configs/
COPY agents/ /app/agents/
COPY tests/ /app/tests/
COPY pyproject.toml /app/

# Create mount points for data, models, reports, mission outputs and the MLflow
# tracking store. Each is bind-mounted by docker-compose so results survive the
# container.
RUN mkdir -p /app/data /app/models /app/reports /app/missions /app/mlruns /app/state

# Default command: run unit tests to verify the installation. The georeferencing
# tests that need real Sentinel-1 products skip cleanly when data/ is empty, so
# this passes on a bare image.
CMD ["pytest", "tests/", "-q", "-m", "not slow"]
