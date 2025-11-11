# RCL Prediction Analysis

This directory contains scripts for running comprehensive experiments and generating comparison tables.

## Quick Start

### Run All Experiments

```bash
bash analysis/run_experiments.sh
```

This will:
1. Pre-compute ESM2 embeddings if needed (can take 1-2 hours)
2. Train models with all combinations:
   - Encodings: one-hot, BLOSUM62, ESM2-650M
   - Models: CNN, U-Net
   - Total: 6 experiments (3 encodings × 2 models)
3. Generate a summary table with validation metrics
4. Save results to `analysis/results/`

### Generate Summary from Existing Results

If you've already run experiments and just want to regenerate the summary:

```bash
python analysis/generate_summary.py
```

**The summary script will:**
- Process all completed experiments in `analysis/results/`
- Skip incomplete runs (no training_history.json)
- Report which experiments are missing
- Generate summary even with partial results (e.g., 4/6 completed)
- Create `analysis/results/summary.txt` and `analysis/results/summary.csv`

## Configuration

Edit `analysis/run_experiments.sh` to change:

- `EPOCHS`: Number of training epochs (default: 50)
- `BATCH_SIZE`: Batch size per GPU (default: 32)
- `ENCODINGS`: Array of encodings to test
- `MODELS`: Array of models to test

Example:
```bash
EPOCHS=100
BATCH_SIZE=16
ENCODINGS=("onehot" "blosum")  # Skip ESM2 for faster testing
MODELS=("cnn" "unet" "lstm")   # Add LSTM model
```

## Output Files

### Summary Table (`analysis/results/summary.txt`)

Example output:
```
========================================
RCL Prediction Results Summary
========================================

Encoding        Model    Accuracy Precision Recall   F1       F1-Macro MCC      AUC     
--------------- -------- -------- --------- -------- -------- -------- -------- --------
onehot          CNN      0.9234   0.8567    0.8912   0.8736   0.9102   0.7845   0.9456  
onehot          UNET     0.9456   0.8934    0.9123   0.9027   0.9345   0.8234   0.9678  
blosum          CNN      0.9312   0.8723    0.8989   0.8854   0.9201   0.7967   0.9523  
blosum          UNET     0.9523   0.9012    0.9234   0.9121   0.9423   0.8412   0.9712  
esm2_650m       CNN      0.9589   0.9234    0.9345   0.9289   0.9534   0.8656   0.9789  
esm2_650m       UNET     0.9678   0.9456    0.9512   0.9483   0.9678   0.8912   0.9856  

Best Overall Performance:
  Encoding: esm2_650m
  Model: UNET
  F1 Score: 0.9483
  Accuracy: 0.9678
  MCC: 0.8912
```

### Individual Results

Each experiment saves to `analysis/results/{encoding}_{model}/`:
- `training_history.json`: Loss and metrics per epoch
- `best_model.pt`: Best model checkpoint
- `final_model.pt`: Final model checkpoint
- `config.json`: Experiment configuration
- `training_curves.png`: Training/validation curves
- `tensorboard/`: TensorBoard logs

### CSV Export (`analysis/results/summary.csv`)

Metrics exported to CSV for easy import into spreadsheets or plotting tools.

## Running Individual Experiments

Instead of running all experiments, you can run individual ones:

```bash
# One-hot + CNN
python src/train.py --encoding onehot --model cnn --epochs 50 --batch-size 32 --multi-gpu

# BLOSUM + U-Net
python src/train.py --encoding blosum --model unet --epochs 50 --batch-size 32 --multi-gpu

# ESM2 + CNN (using precomputed embeddings)
python src/train.py --encoding esm2_650m --model cnn --epochs 50 --batch-size 8 \
    --precomputed data/embeddings/esm2_650m_embeddings.npz --multi-gpu

# ESM2 + U-Net (using precomputed embeddings)
python src/train.py --encoding esm2_650m --model unet --epochs 50 --batch-size 8 \
    --precomputed data/embeddings/esm2_650m_embeddings.npz --multi-gpu
```

**Or run ESM2 experiments together:**

```bash
bash analysis/run_esm2_experiments.sh
```

This script runs experiments 5 and 6 (ESM2 with both models) separately from the full suite.

## Tips

1. **ESM2 Embeddings**: Pre-compute once, use many times
   ```bash
   python src/precompute_embeddings.py --encoding esm2_650m --batch-size 1 --checkpoint-every 500
   ```

2. **Resume if Interrupted**: The precompute script supports resuming
   ```bash
   python src/precompute_embeddings.py --encoding esm2_650m --resume
   ```

3. **Quick Testing**: Use fewer epochs and simpler encodings
   ```bash
   # Edit run_experiments.sh
   EPOCHS=10
   ENCODINGS=("onehot" "blosum")
   ```

4. **Monitor Progress**: Use TensorBoard
   ```bash
   tensorboard --logdir analysis/results/
   ```

## Expected Runtime

- One-hot + CNN: ~30 min
- One-hot + U-Net: ~40 min
- BLOSUM + CNN: ~30 min
- BLOSUM + U-Net: ~40 min
- ESM2 + CNN: ~40 min (if embeddings precomputed)
- ESM2 + U-Net: ~50 min (if embeddings precomputed)

**Total**: ~4 hours for all experiments (+ 1-2 hours for ESM2 precomputation)

## Troubleshooting

**Out of Memory**: Reduce batch size in the script
```bash
BATCH_SIZE=16  # or even 8
```

**ESM2 Killed**: Use checkpointing
```bash
python src/precompute_embeddings.py --encoding esm2_650m --checkpoint-every 100 --resume
```

**Results Missing**: Check `runs/` directory for failed experiments
```bash
ls -ltr runs/
```
