#!/bin/bash
# Training script for Energy Prediction System
# This script automates the model training pipeline

set -e  # Exit on error

# Display help
show_help() {
  echo "Usage: $0 [options]"
  echo
  echo "Options:"
  echo "  -d, --data PATH       Path to training data (required)"
  echo "  -c, --config PATH     Path to config file (default: configs/model_config.yaml)"
  echo "  -o, --output DIR      Output directory for models (default: models/trained)"
  echo "  -p, --plot            Generate plots"
  echo "  -v, --verbose         Enable verbose output"
  echo "  -h, --help            Display this help message"
  echo
}

# Default values
DATA_PATH=""
CONFIG_PATH="configs/model_config.yaml"
OUTPUT_DIR="models/trained"
GENERATE_PLOTS=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--data)
      DATA_PATH="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -o|--output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -p|--plot)
      GENERATE_PLOTS=true
      shift
      ;;
    -v|--verbose)
      VERBOSE=true
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

# Check required arguments
if [ -z "$DATA_PATH" ]; then
  echo "Error: Data path is required"
  show_help
  exit 1
fi

# Check if files exist
if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Data file not found: $DATA_PATH"
  exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "Error: Config file not found: $CONFIG_PATH"
  exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "=============================================="
echo "Energy Prediction System - Model Training"
echo "=============================================="
echo "Data path: $DATA_PATH"
echo "Config path: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Generate plots: $GENERATE_PLOTS"
echo "Log file: $LOG_FILE"
echo "=============================================="

# Build command
CMD="python -m scripts.train"
CMD="$CMD --config $CONFIG_PATH"
CMD="$CMD --data $DATA_PATH"
CMD="$CMD --output $OUTPUT_DIR"

if [ "$GENERATE_PLOTS" = true ]; then
  CMD="$CMD --plot"
fi

if [ "$VERBOSE" = true ]; then
  CMD="$CMD --verbose"
fi

# Run training script
echo "Starting training process..."
echo "Command: $CMD"
echo "This may take some time. Check the log file for progress."

if [ "$VERBOSE" = true ]; then
  # Run with output to both console and log file
  $CMD 2>&1 | tee "$LOG_FILE"
else
  # Run with output to log file only
  $CMD > "$LOG_FILE" 2>&1
fi

# Check if training was successful
if [ $? -eq 0 ]; then
  echo "Training completed successfully!"
  echo "Models saved to: $OUTPUT_DIR"
  echo "Log file: $LOG_FILE"
else
  echo "Training failed. Check log file for details: $LOG_FILE"
  exit 1
fi

# Run evaluation if training was successful
if [ "$GENERATE_PLOTS" = true ]; then
  echo "Generating evaluation plots..."
  python -m src.visualization.visualize --model-dir "$OUTPUT_DIR" --output-dir "results/plots"
fi

echo "=============================================="
echo "Training process complete"
echo "=============================================="
