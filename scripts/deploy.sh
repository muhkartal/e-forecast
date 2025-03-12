#!/bin/bash
# Deployment script for Energy Prediction System
# This script automates the deployment of the API service

set -e  # Exit on error

# Display help
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -e, --env ENV         Environment (dev, staging, prod) (default: dev)"
  echo "  -p, --port PORT       Port for the API service (default: 8000)"
  echo "  -m, --models DIR      Directory containing trained models (default: models/trained)"
  echo "  -c, --config PATH     Path to API config file (default: configs/api_config.yaml)"
  echo "  -d, --docker          Deploy using Docker (default: true)"
  echo "  -s, --scale N         Number of API instances to run (default: 1)"
  echo "  -r, --reload          Enable auto-reload for development (default: false)"
  echo "  -h, --help            Display this help message"
  echo
}

# Default values
ENVIRONMENT="dev"
PORT=8000
MODELS_DIR="models/trained"
CONFIG_PATH="configs/api_config.yaml"
USE_DOCKER=true
SCALE=1
AUTO_RELOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -m|--models)
      MODELS_DIR="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -d|--docker)
      if [[ "$2" == "false" ]]; then
        USE_DOCKER=false
      fi
      shift 2
      ;;
    -s|--scale)
      SCALE="$2"
      shift 2
      ;;
    -r|--reload)
      AUTO_RELOAD=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Error: Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# Check if files exist
if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: Models directory not found: $MODELS_DIR"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

# Create logs directory
mkdir -p logs

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/deployment_${TIMESTAMP}.log"

echo "=============================================="
echo "Energy Prediction System - Deployment"
echo "=============================================="
echo "Environment: $ENVIRONMENT"
echo "Port: $PORT"
echo "Models directory: $MODELS_DIR"
echo "Config path: $CONFIG_PATH"
echo "Using Docker: $USE_DOCKER"
echo "Scale: $SCALE"
echo "Auto-reload: $AUTO_RELOAD"
echo "Log file: $LOG_FILE"
echo "=============================================="

# Set environment variables
export CONFIG_PATH="$CONFIG_PATH"
export MODEL_DIR="$MODELS_DIR"
export PORT="$PORT"
export ENVIRONMENT="$ENVIRONMENT"

# Docker deployment
if [ "$USE_DOCKER" = true ]; then
  echo "Deploying using Docker..."

  # Check if Docker is installed
  if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
  fi

  # Check if Docker Compose is installed
  if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    exit 1
  fi

  # Create .env file for Docker Compose
  echo "Creating .env file for Docker Compose..."
  cat > .env << EOF
CONFIG_PATH=${CONFIG_PATH}
MODEL_DIR=${MODELS_DIR}
PORT=${PORT}
ENVIRONMENT=${ENVIRONMENT}
SCALE=${SCALE}
EOF

  # Build and start containers
  echo "Building and starting containers..."
  if [ "$ENVIRONMENT" = "prod" ]; then
    # Production deployment
    docker-compose -f docker/docker-compose.yml up -d --build --scale api=${SCALE}
  else
    # Development deployment
    docker-compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d --build --scale api=${SCALE}
  fi

  # Check if deployment was successful
  if [ $? -eq 0 ]; then
    echo "Docker deployment completed successfully!"
    echo "The API service is running at: http://localhost:${PORT}"
    echo "Monitoring available at:"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
  else
    echo "Deployment failed. Check logs for details."
    exit 1
  fi

# Local deployment
else
  echo "Deploying locally..."

  # Check if Python is installed
  if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
  fi

  # Check if the virtual environment exists
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
  fi

  # Activate virtual environment
  echo "Activating virtual environment..."
  source venv/bin/activate

  # Install dependencies
  echo "Installing dependencies..."
  pip install -r requirements.txt

  # Start the API service
  echo "Starting API service..."

  # Build command
  CMD="uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"

  if [ "$AUTO_RELOAD" = true ]; then
    CMD="$CMD --reload"
  fi

  # Run API service
  echo "Command: $CMD"
  $CMD > "$LOG_FILE" 2>&1 &

  # Save the process ID
  PID=$!
  echo $PID > .api.pid

  echo "API service started with PID: $PID"
  echo "The API service is running at: http://localhost:${PORT}"
  echo "Documentation available at: http://localhost:${PORT}/docs"
  echo "To stop the service, run: kill $(cat .api.pid)"
fi

echo "=============================================="
echo "Deployment process complete"
echo "=============================================="
