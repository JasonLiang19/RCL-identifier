# RCL Predictor - PyTorch Implementation

A modern PyTorch-based tool for predicting Reactive Center Loop (RCL) locations in serpin protein sequences.

## Features

- **Multiple Encoding Schemes**: 
  - One-hot (21-dimensional)
  - BLOSUM62 (20-dimensional)
  - ESM2 HuggingFace (1280-dimensional, on-the-fly)
  - ESM2_650M native fair-esm (1280-dimensional, pre-computed recommended)
- **Multiple Architectures**: CNN, U-Net with attention gates, LSTM
- **Pre-computation Support**: Checkpoint-based embedding generation for large models
- **Multi-GPU Training**: DataParallel support for faster training
- **PyTorch-native**: Built entirely with PyTorch 2.x for flexibility and performance
- **FASTA I/O**: Read protein sequences and output predictions in FASTA format
- **Experiment Tracking**: Organized model checkpoints and metrics with TensorBoard
- **Modular Design**: Easy to add new encodings or architectures

## Improvements Over Previous Version

1. **Modern Framework**: Migrated from TensorFlow 2.9 to PyTorch 2.x
2. **Flexible Architecture**: Modular design allows easy testing of different models
3. **Better Encoding Support**: Simplified ESM2 integration, removed deprecated bio-embeddings
4. **Cleaner Code**: Separated concerns (data, models, training, inference)
5. **Better Metrics**: Enhanced visualization and performance tracking
6. **Production Ready**: Easy-to-use inference script for annotating new sequences

## Installation

```bash
# Create conda environment
conda create -n rcl-id python=3.10
conda activate rcl-id

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Pre-computing ESM2_650M Embeddings (Recommended)

For large datasets, pre-compute ESM2 embeddings to avoid memory issues during training:

```bash
# Pre-compute embeddings with checkpointing (saves every 200 sequences)
python src/precompute_embeddings.py --encoding esm2_650m --checkpoint-every 200

# Resume from checkpoint if interrupted
python src/precompute_embeddings.py --encoding esm2_650m --resume

# Verify embeddings were created
ls data/embeddings/esm2_650m_embeddings.npz
```

**Why pre-compute?**
- ESM2_650M model is very large (~2GB) and requires significant GPU memory
- Encoding 3000+ sequences on-the-fly can cause OOM (Out of Memory) errors
- Pre-computing once allows unlimited training experiments without re-encoding
- Checkpointing enables resume after interruptions (Ctrl+C, SLURM time limits, etc.)

**Storage**: Pre-computed embeddings require ~500MB-1GB disk space.

### Training

```bash
# With pre-computed embeddings (recommended for ESM2)
python src/train.py --encoding esm2_650m --model unet --precomputed --epochs 50

# Without pre-computed embeddings (for smaller encodings)
python src/train.py --encoding onehot --model cnn --epochs 50 --batch-size 32

# Multi-GPU training (if you have multiple GPUs)
python src/train.py --encoding blosum --model unet --multi-gpu --batch-size 64
```

### Inference

```bash
python src/predict.py --input sequences.fasta --output predictions.fasta --model-dir runs/run_001
```

### Evaluation

```bash
python src/evaluate.py --test-file data/test.csv --model-dir runs/run_001
```

## Project Structure

```
RCL/
├── data/                    # Data files
│   ├── embeddings/         # Pre-computed embeddings (ESM2)
│   ├── encodings/          # Encoding matrices (BLOSUM62, One-hot)
│   ├── raw/                # Raw CSV files with annotations
│   ├── scripts/            # Data validation and cleaning tools
│   └── validation_reports/ # Data quality reports
├── src/                    # Source code
│   ├── data/               # Data loading and encoding
│   │   ├── encoders.py    # OneHot, BLOSUM, ESM2 encoders
│   │   └── data_loader.py # CSV/FASTA data loading
│   ├── models/             # Model architectures
│   │   └── architectures.py # CNN, U-Net, LSTM models
│   ├── utils/              # Utilities and metrics
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   ├── predict.py          # Inference script
│   └── precompute_embeddings.py # ESM2 pre-computation
├── analysis/               # Experiment automation
│   ├── run_experiments.sh # Run all encoding/model combinations
│   └── generate_summary.py # Aggregate results
├── runs/                   # Experiment outputs (checkpoints, logs)
├── requirements.txt
├── config.yaml            # Configuration file
└── README.md
```

## Usage Examples

### Pre-compute Embeddings for ESM2

```bash
# First time: Pre-compute ESM2_650M embeddings
python src/precompute_embeddings.py --encoding esm2_650m --checkpoint-every 500

# If interrupted, resume from last checkpoint
python src/precompute_embeddings.py --encoding esm2_650m --resume

# Check progress (if checkpoint exists)
python -c "import numpy as np; d=np.load('data/embeddings/esm2_650m_embeddings_checkpoint.npz'); print(f'Progress: {len(d[\"ids\"])} sequences encoded')"
```

### Train with Different Configurations

```bash
# CNN with BLOSUM62 encoding
python src/train.py --encoding blosum --model cnn

# U-Net with ESM2 embeddings (using pre-computed)
python src/train.py --encoding esm2_650m --model unet --precomputed

# U-Net with on-the-fly ESM2 (HuggingFace version, may cause OOM)
python src/train.py --encoding esm2 --model unet

# One-hot encoding with LSTM
python src/train.py --encoding onehot --model lstm --epochs 50
```

### Run Complete Experiment Suite

```bash
# Automatically runs all encoding/model combinations
# Pre-computes ESM2 if needed, then trains 6 experiments
cd analysis
bash run_experiments.sh

# View results summary
cat results/summary.csv
```

### Annotate Protein Sequences

```bash
python src/predict.py \
    --input my_proteins.fasta \
    --output annotated.fasta \
    --model-dir runs/best_model
```

Output format:
```
>ProteinID rcl:350-370 score:0.95
MYLKIVILVTFPLVCFTQDDTPL...
```

## Data Format

### Training Data (CSV)
Columns: `id`, `Sequence`, `rcl_start`, `rcl_end`, `rcl_seq`

### Input FASTA
```
>ProteinID
SEQUENCEHERE...
```

### Output FASTA
```
>ProteinID rcl:start-end score:confidence
SEQUENCEHERE...
```

## Model Architectures

1. **CNN**: Multi-layer 1D convolutional network with batch normalization, dropout, and residual connections
2. **U-Net**: Encoder-decoder architecture with attention gates for precise sequence segmentation
3. **LSTM**: Bidirectional LSTM for capturing long-range sequence dependencies

## Encoding Schemes

1. **One-hot**: Traditional 21-dimensional encoding (20 amino acids + unknown)
2. **BLOSUM62**: Evolutionary substitution matrix (20-dimensional)
3. **ESM2 (HuggingFace)**: Protein language model using transformers library (1280-dim, on-the-fly)
4. **ESM2_650M (native)**: Facebook's fair-esm implementation (1280-dim, pre-computed recommended)

**Recommendation**: Use ESM2_650M with pre-computation for best results on large datasets.

## HPC/SLURM Usage

For HiPerGator or other SLURM clusters:

```bash
# Interactive session with GPU
srun --partition=gpu --gpus=1 --mem=32gb --time=4:00:00 --pty bash

# Load modules
module load conda
conda activate rcl-id

# Pre-compute embeddings (can take 1-2 hours)
python src/precompute_embeddings.py --encoding esm2_650m --checkpoint-every 500

# Submit batch job for training
sbatch your_training_script.sbatch
```

Example SBATCH script:
```bash
#!/bin/bash
#SBATCH --job-name=rcl_train
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --mem=64gb
#SBATCH --time=24:00:00

module load conda
conda activate rcl-id

# Train with pre-computed embeddings and multi-GPU
python src/train.py --encoding esm2_650m --model unet --precomputed --multi-gpu --epochs 100
```

## Troubleshooting

### Out of Memory (OOM) Errors

**Problem**: Training crashes with CUDA OOM or process killed.

**Solutions**:
1. **For ESM2 models**: Always pre-compute embeddings first
   ```bash
   python src/precompute_embeddings.py --encoding esm2_650m --checkpoint-every 500
   python src/train.py --encoding esm2_650m --model unet --precomputed
   ```
2. Reduce batch size: `--batch-size 16` or `--batch-size 8`
3. Use gradient accumulation (if implemented)

### Data Validation Issues

**Problem**: Training fails with "invalid literal for int()" or annotation errors.

**Solution**: Validate your CSV files before training:
```bash
python data/scripts/data_cleaning.py --input "data/raw/Alphafold_RCL_annotations.csv"
```

This will identify rows with:
- Missing or invalid RCL annotations ('no result', 'no start', etc.)
- Sequence/coordinate mismatches
- Invalid amino acid characters

### Pre-computation Interrupted

**Problem**: ESM2 pre-computation stopped partway (Ctrl+C, SLURM timeout, etc.)

**Solution**: Resume from checkpoint:
```bash
python src/precompute_embeddings.py --encoding esm2_650m --resume
```

Checkpoints are saved every 500 sequences by default.

### Multi-GPU Not Detected

**Problem**: Training doesn't use multiple GPUs.

**Solution**:
1. Verify GPUs are available: `nvidia-smi`
2. Use the `--multi-gpu` flag explicitly
3. Check PyTorch sees GPUs: `python -c "import torch; print(torch.cuda.device_count())"`

## Citation

If you use this software, please cite the original work and acknowledge this implementation.

## License

[Specify your license here]
