#!/bin/bash
# Setup script for RCL Predictor

set -e

echo "=========================================="
echo "RCL Predictor Setup"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p runs
mkdir -p models
mkdir -p predictions
mkdir -p data/raw
mkdir -p notebooks

echo "✓ Directories created"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo "✓ Dependencies installed"

# Check for GPU
echo ""
echo "Checking for GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review config.yaml and adjust settings"
echo "2. Run a test training: python src/train.py --encoding onehot --model cnn --epochs 5"
echo "3. See USAGE.md for more examples"
echo ""
