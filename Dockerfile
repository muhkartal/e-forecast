# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

# Set up timezone
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone

# Install essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel

# Stage for building dependencies
FROM base AS dependencies

# Install Poetry
RUN pip install poetry==1.4.2

# Set up poetry to not use virtualenvs
RUN poetry config virtualenvs.create false

# Copy only dependency files
WORKDIR /app
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Final stage
FROM base AS final

# Copy installed dependencies from previous stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set up app directory
WORKDIR /app

# Copy project files
COPY . .

# Create directories for data, models, and logs
RUN mkdir -p data/raw data/processed models/trained logs configs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create configuration link
RUN ln -sf /app/configs/api_config.yaml /app/src/api/config.yaml

# Expose port for API
EXPOSE 8000

# Set environment variables for runtime
ENV PYTHONPATH=/app \
    CONFIG_PATH=/app/configs/api_config.yaml \
    LOG_LEVEL=INFO

# Command to run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
