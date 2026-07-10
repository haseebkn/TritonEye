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
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy configuration and agent modules
COPY configs/ /app/configs/
COPY agents/ /app/agents/

# Create mount points for data, models, reports, and mission run logs
RUN mkdir -p /app/data /app/models /app/reports /app/missions

# Default command: run unit tests to verify the installation
CMD ["pytest", "tests/"]
