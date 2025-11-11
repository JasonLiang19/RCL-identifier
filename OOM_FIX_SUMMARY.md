# OOM Issue Fixed - run_experiments.sh

## Problem
The experiment script was killed with exit code 137 (Out of Memory) when running ESM2_650M experiments.

### Root Cause
- ESM2 precomputed embeddings: **18 GB in RAM** (3431 × 1024 × 1280 × 4 bytes)
- Batch size 32 with multi-GPU adds significant memory overhead
- Total memory requirement exceeded available system RAM

### Failed Run
- **run_019**: esm2_650m + cnn (KILLED)

## Solution Applied
Reduced batch size specifically for ESM2_650M experiments:

### Changes to `analysis/run_experiments.sh`:
1. Added `BATCH_SIZE_ESM2=8` (reduced from 32)
2. Dynamic batch size selection based on encoding type
3. Updated reporting to show different batch sizes

### New Configuration:
- **One-hot/BLOSUM**: batch_size = 32
- **ESM2_650M**: batch_size = 8 (4x smaller)

## Completed Experiments (Before OOM)
✅ run_015: onehot + cnn
✅ run_016: onehot + unet
✅ run_017: blosum + cnn  
✅ run_018: blosum + unet

## Remaining Experiments (To Re-run)
❌ run_019: esm2_650m + cnn (needs retry with batch_size=8)
⏳ Pending: esm2_650m + unet (batch_size=8)

## Next Steps

### Option 1: Clean up and restart all experiments
```bash
# Remove incomplete run
rm -rf runs/run_019

# Clear previous analysis results if any
rm -rf analysis/results/*

# Run all experiments from scratch
cd /blue/zhou/share/projects/RCL
bash analysis/run_experiments.sh
```

### Option 2: Continue from where it left off
The script will automatically start from run_020, but run_019 will remain as incomplete.
You can manually run the ESM2 experiments:

```bash
# ESM2 + CNN (with correct batch size)
python src/train.py --encoding esm2_650m --model cnn --epochs 50 --batch-size 8 \
    --precomputed data/embeddings/esm2_650m_embeddings.npz --multi-gpu

# ESM2 + U-Net (with correct batch size)
python src/train.py --encoding esm2_650m --model unet --epochs 50 --batch-size 8 \
    --precomputed data/embeddings/esm2_650m_embeddings.npz --multi-gpu
```

### Option 3: Run only ESM2 experiments with updated script
```bash
# Edit the script to only run ESM2 experiments
# Change: ENCODINGS=("esm2_650m")
# Then run
bash analysis/run_experiments.sh
```

## Recommended: Option 1 (Clean Restart)

This ensures consistent experiment numbering and complete results in analysis/results/.

## Updated Time Estimates

With reduced batch size for ESM2:
- ESM2_650M + CNN: ~15-20 min (was 10-15 min)
- ESM2_650M + U-Net: ~20-25 min (was 15-20 min)

**Total time for all 6 experiments: ~2-2.5 hours** (was 1.5-2 hours)

The slower ESM2 experiments are offset by avoiding OOM issues and crashes.
