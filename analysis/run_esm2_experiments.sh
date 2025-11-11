#!/bin/bash

# Run ESM2 experiments separately (experiments 5 and 6)
# This script runs the ESM2 encoding experiments with both CNN and U-Net models

set -e  # Exit on error

# Configuration
ESM2_EMBEDDINGS="data/embeddings/esm2_650m_embeddings.npz"
BATCH_SIZE=8
EPOCHS=50

echo "=========================================="
echo "Running ESM2 Experiments (5/6 and 6/6)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - Embeddings: ${ESM2_EMBEDDINGS}"
echo "  - Batch size: ${BATCH_SIZE}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Multi-GPU: enabled"
echo ""

# Check if embeddings exist
if [ ! -f "${ESM2_EMBEDDINGS}" ]; then
    echo "Error: ESM2 embeddings not found at ${ESM2_EMBEDDINGS}"
    exit 1
fi

# Experiment 5/6: ESM2 + CNN
echo "=========================================="
echo "Experiment 5/6: esm2_650m + CNN"
echo "=========================================="
python src/train.py \
    --encoding esm2_650m \
    --model cnn \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --precomputed ${ESM2_EMBEDDINGS} \
    --multi-gpu

echo ""
echo "Experiment 5/6 completed successfully!"
echo ""

# Experiment 6/6: ESM2 + U-Net
echo "=========================================="
echo "Experiment 6/6: esm2_650m + U-Net"
echo "=========================================="
python src/train.py \
    --encoding esm2_650m \
    --model unet \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --precomputed ${ESM2_EMBEDDINGS} \
    --multi-gpu

echo ""
echo "Experiment 6/6 completed successfully!"
echo ""

echo "=========================================="
echo "All ESM2 experiments completed!"
echo "=========================================="
echo ""
echo "To generate summary with all results, run:"
echo "  python analysis/generate_summary.py"
