# RCL Predictor - Quick Reference

## Installation
```bash
cd /blue/zhou/share/projects/RCL
pip install -r requirements.txt
python check_setup.py  # Verify installation
```

## Training

### Basic Training
```bash
python src/train.py --encoding onehot --model cnn
```

### Common Options
```bash
python src/train.py \
    --encoding [onehot|blosum|esm2] \
    --model [cnn|unet|lstm] \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --device cuda \
    --multi-gpu          # Use all available GPUs
    --gpu-ids 0,1        # Use specific GPUs
```

### Training Examples
```bash
# Fast training (one-hot + CNN)
python src/train.py --encoding onehot --model cnn --epochs 20

# Best performance (BLOSUM + U-Net)
python src/train.py --encoding blosum --model unet --epochs 50

# Multi-GPU training (2x B200 GPUs, recommended!)
python src/train.py --encoding blosum --model unet --epochs 50 --multi-gpu --batch-size 64

# Highest quality (ESM2 + LSTM, requires large GPU)
python src/train.py --encoding esm2 --model lstm --batch-size 8 --epochs 30

# Multi-GPU with ESM2
python src/train.py --encoding esm2 --model unet --batch-size 16 --multi-gpu --epochs 30
```

## Evaluation

### Evaluate on Test Set
```bash
python src/evaluate.py \
    --model-dir runs/run_001 \
    --test-file ../rcl-unet/data/Uniprot_Test_Set.csv
```

### Compare Models
```bash
# Evaluate multiple runs
for run in runs/run_*; do
    python src/evaluate.py --model-dir $run --test-file test.csv
done
```

## Prediction (Inference)

### Annotate FASTA File
```bash
python src/predict.py \
    --input sequences.fasta \
    --output predictions.fasta \
    --model-dir runs/run_001
```

### With CSV Output
```bash
python src/predict.py \
    --input sequences.fasta \
    --output predictions.fasta \
    --model-dir runs/run_001 \
    --csv predictions.csv \
    --threshold 0.5
```

## Configuration

### Edit Config File
```bash
nano config.yaml  # or vim, emacs, etc.
```

### Key Settings to Adjust
- `data.max_length`: Maximum sequence length (default: 1024)
- `training.batch_size`: Batch size (default: 32)
- `training.learning_rate`: Learning rate (default: 0.001)
- `training.epochs`: Number of epochs (default: 50)
- `training.loss_weights`: Class weights for imbalance (default: [1.0, 5.0])

## Monitoring

### View Training Progress
```bash
# TensorBoard
tensorboard --logdir runs/

# Training logs
tail -f runs/run_XXX/tensorboard/events.out.tfevents.*

# Metrics
cat runs/run_XXX/config.json
cat runs/run_XXX/training_history.json
```

### Check Results
```bash
# View metrics
cat runs/run_XXX/evaluation/metrics.json

# View predictions
head runs/run_XXX/evaluation/predictions.csv
```

## Output Formats

### Training Output Structure
```
runs/run_XXX/
├── best_model.pt           # Best model checkpoint
├── final_model.pt          # Final model checkpoint
├── config.json             # Run configuration
├── training_history.json   # Training metrics
├── loss_curves.png         # Loss plots
├── metric_curves.png       # Metric plots
└── tensorboard/            # TensorBoard logs
```

### Evaluation Output Structure
```
runs/run_XXX/evaluation/
├── metrics.json            # Performance metrics
├── predictions.csv         # Detailed predictions
├── confusion_matrix.png    # Confusion matrix
├── error_histogram.png     # Error distribution
└── error_counts.csv        # Error statistics
```

### FASTA Output Format
```
>ProteinID rcl:350-370 score:0.95
SEQUENCEHERE...

>AnotherProtein rcl:not_identified score:0.00
ANOTHERSEQUENCE...
```

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python src/train.py --batch-size 16

# Use smaller model
python src/train.py --model cnn

# Use simpler encoding
python src/train.py --encoding onehot

# If using multi-GPU, batch size is per GPU
python src/train.py --multi-gpu --batch-size 32  # Total: 64 with 2 GPUs
```

### Poor Performance
```bash
# Try different architecture
python src/train.py --model unet

# Try different encoding
python src/train.py --encoding blosum

# Adjust loss weights in config.yaml
training:
  loss_weights: [1.0, 10.0]  # Increase RCL weight
```

### Slow Training
```bash
# Use GPU
python src/train.py --device cuda

# Reduce workers if I/O bound
# Edit config.yaml:
hardware:
  num_workers: 2
```

## File Locations

### Data Files (from old project)
- Training: `../rcl-unet/data/Alphafold_RCL_annotations.csv`
- Training (non-serpin): `../rcl-unet/data/non_serpin_train.csv`
- Test: `../rcl-unet/data/Uniprot_Test_Set.csv`
- Test (non-serpin): `../rcl-unet/data/non_serpin_test.csv`
- FASTA: `../rcl-unet/data/RCL_annotations.fasta`

### Encodings
- One-hot: `data/encodings/One_hot.json`
- BLOSUM62: `data/encodings/BLOSUM62.json`

## Common Workflows

### 1. Train and Evaluate
```bash
# Train
python src/train.py --encoding blosum --model unet

# Evaluate
python src/evaluate.py --model-dir runs/run_001 --test-file test.csv
```

### 2. Compare Encodings
```bash
python src/train.py --encoding onehot --model cnn
python src/train.py --encoding blosum --model cnn
python src/train.py --encoding esm2 --model cnn --batch-size 8
```

### 3. Annotate New Sequences
```bash
# Get sequences (example)
cat > my_sequences.fasta << EOF
>Protein1
MYLKIVILVTFPLVCFTQDDTPLSKPMAIDYQAEFAW
>Protein2
MQGLKMRFLAPLGIVLLAIVSHCQGQGFDPTVGATP
EOF

# Predict
python src/predict.py -i my_sequences.fasta -o predictions.fasta -m runs/run_001
```

## Useful Commands

```bash
# List all runs
ls -ltr runs/

# Find best run by F1 score
for run in runs/run_*; do
    echo -n "$run: "
    grep '"val_f1"' $run/training_history.json | tail -1
done

# Count sequences in FASTA
grep -c "^>" sequences.fasta

# Check model size
du -sh runs/run_*/best_model.pt

# Clean old runs (careful!)
# rm -rf runs/run_00{1..5}
```

## Help

```bash
python src/train.py --help
python src/evaluate.py --help
python src/predict.py --help
```

## Documentation

- `README.md` - Project overview
- `USAGE.md` - Detailed usage guide
- `PROJECT_SUMMARY.md` - Complete project documentation
- `MULTI_GPU_GUIDE.md` - **Multi-GPU training guide (2x B200 setup)**
- `QUICK_REFERENCE.md` - This file
- `examples.py` - Code examples
