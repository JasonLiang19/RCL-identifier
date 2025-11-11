#!/bin/bash
#
# Run experiments with different encodings and models
# Usage: bash analysis/run_experiments.sh
#

set -e  # Exit on error

# Configuration
EPOCHS=50
BATCH_SIZE=32
BATCH_SIZE_ESM2=8  # Smaller batch size for ESM2 due to large embedding size
RESULTS_DIR="analysis/results"
SUMMARY_FILE="analysis/results/summary.txt"

# Create results directory
mkdir -p ${RESULTS_DIR}

# Arrays for encodings and models
ENCODINGS=("onehot" "blosum" "esm2_650m")
MODELS=("cnn" "unet")

echo "=========================================="
echo "RCL Prediction Experiments"
echo "=========================================="
echo "Encodings: ${ENCODINGS[@]}"
echo "Models: ${MODELS[@]}"
echo "Epochs: ${EPOCHS}"
echo "Batch Size (onehot/blosum): ${BATCH_SIZE}"
echo "Batch Size (esm2_650m): ${BATCH_SIZE_ESM2}"
echo "=========================================="
echo ""

# Clear previous summary
> ${SUMMARY_FILE}

# Function to extract best validation metrics from training history
extract_metrics() {
    local run_dir=$1
    local history_file="${run_dir}/training_history.json"
    
    if [ ! -f "${history_file}" ]; then
        echo "N/A N/A N/A N/A N/A N/A N/A"
        return
    fi
    
    # Use Python to extract best metrics (at epoch with best F1)
    python3 << EOF
import json
import numpy as np

with open('${history_file}', 'r') as f:
    history = json.load(f)

# Find epoch with best F1
val_metrics = history['val_metrics']
f1_scores = val_metrics['f1']
best_epoch = np.argmax(f1_scores)

# Extract all metrics at best epoch
metrics = {}
for metric in ['accuracy', 'precision', 'recall', 'f1', 'f1_macro', 'mcc', 'auc']:
    if metric in val_metrics:
        metrics[metric] = val_metrics[metric][best_epoch]
    else:
        metrics[metric] = 0.0

# Print in order
print(f"{metrics['accuracy']:.4f} {metrics['precision']:.4f} {metrics['recall']:.4f} {metrics['f1']:.4f} {metrics['f1_macro']:.4f} {metrics['mcc']:.4f} {metrics['auc']:.4f}")
EOF
}

# Check if ESM2 embeddings need to be precomputed
if [[ " ${ENCODINGS[@]} " =~ " esm2_650m " ]]; then
    ESM2_EMBEDDINGS="data/embeddings/esm2_650m_embeddings.npz"
    if [ ! -f "${ESM2_EMBEDDINGS}" ]; then
        echo "⚠ ESM2 embeddings not found. Precomputing..."
        echo "This may take 1-2 hours..."
        python src/precompute_embeddings.py --encoding esm2_650m --batch-size 1 --checkpoint-every 500
        echo "✓ ESM2 embeddings precomputed"
        echo ""
    else
        echo "✓ ESM2 embeddings found at ${ESM2_EMBEDDINGS}"
        echo ""
    fi
fi

# Run experiments
experiment_count=0
total_experiments=$((${#ENCODINGS[@]} * ${#MODELS[@]}))

for encoding in "${ENCODINGS[@]}"; do
    for model in "${MODELS[@]}"; do
        experiment_count=$((experiment_count + 1))
        
        echo ""
        echo "=========================================="
        echo "Experiment ${experiment_count}/${total_experiments}"
        echo "Encoding: ${encoding} | Model: ${model}"
        echo "=========================================="
        
        # Set batch size based on encoding
        if [ "${encoding}" == "esm2_650m" ]; then
            current_batch_size=${BATCH_SIZE_ESM2}
        else
            current_batch_size=${BATCH_SIZE}
        fi
        
        # Build command
        if [ "${encoding}" == "esm2_650m" ]; then
            # Use precomputed embeddings for ESM2
            cmd="python src/train.py \
                --encoding esm2_650m \
                --model ${model} \
                --epochs ${EPOCHS} \
                --batch-size ${current_batch_size} \
                --precomputed ${ESM2_EMBEDDINGS} \
                --multi-gpu"
        else
            # Encode on-the-fly for one-hot and BLOSUM
            cmd="python src/train.py \
                --encoding ${encoding} \
                --model ${model} \
                --epochs ${EPOCHS} \
                --batch-size ${current_batch_size} \
                --multi-gpu"
        fi
        
        echo "Command: ${cmd}"
        echo ""
        
        # Run training
        eval ${cmd}
        
        # Find the latest run directory
        latest_run=$(ls -td runs/run_* | head -1)
        
        # Copy results to analysis directory
        result_name="${encoding}_${model}"
        cp -r ${latest_run} ${RESULTS_DIR}/${result_name}
        
        echo ""
        echo "✓ Completed: ${encoding} + ${model}"
        echo "✓ Results saved to: ${RESULTS_DIR}/${result_name}"
        echo ""
    done
done

echo ""
echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Generating summary table..."
echo ""

# Generate summary table
cat > ${SUMMARY_FILE} << 'HEADER'
========================================
RCL Prediction Results Summary
========================================

HEADER

echo "Experiment Configuration:" >> ${SUMMARY_FILE}
echo "  Epochs: ${EPOCHS}" >> ${SUMMARY_FILE}
echo "  Batch Size (onehot/blosum): ${BATCH_SIZE}" >> ${SUMMARY_FILE}
echo "  Batch Size (esm2_650m): ${BATCH_SIZE_ESM2}" >> ${SUMMARY_FILE}
echo "  Multi-GPU: Yes (2x B200)" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

echo "Validation Metrics (at best F1 epoch):" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

# Create table header
printf "%-15s %-8s %-8s %-9s %-8s %-8s %-8s %-8s %-8s\n" \
    "Encoding" "Model" "Accuracy" "Precision" "Recall" "F1" "F1-Macro" "MCC" "AUC" >> ${SUMMARY_FILE}
printf "%-15s %-8s %-8s %-9s %-8s %-8s %-8s %-8s %-8s\n" \
    "---------------" "--------" "--------" "---------" "--------" "--------" "--------" "--------" "--------" >> ${SUMMARY_FILE}

# Populate table
for encoding in "${ENCODINGS[@]}"; do
    for model in "${MODELS[@]}"; do
        result_dir="${RESULTS_DIR}/${encoding}_${model}"
        
        # Extract metrics
        metrics=$(extract_metrics ${result_dir})
        read acc prec rec f1 f1_macro mcc auc <<< ${metrics}
        
        # Print row
        printf "%-15s %-8s %-8s %-9s %-8s %-8s %-8s %-8s %-8s\n" \
            "${encoding}" "${model}" "${acc}" "${prec}" "${rec}" "${f1}" "${f1_macro}" "${mcc}" "${auc}" >> ${SUMMARY_FILE}
    done
done

echo "" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}
echo "Individual results saved in: ${RESULTS_DIR}/" >> ${SUMMARY_FILE}
echo "========================================" >> ${SUMMARY_FILE}

# Display summary
cat ${SUMMARY_FILE}

echo ""
echo "✓ Summary saved to: ${SUMMARY_FILE}"
echo ""
echo "To view plots and detailed results, check:"
echo "  ${RESULTS_DIR}/"
echo ""
