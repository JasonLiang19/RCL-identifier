# RCL Predictor - Usage Guide

## Quick Start Examples

### 1. Training a Model

Train a CNN model with one-hot encoding:
```bash
cd /blue/zhou/share/projects/RCL
python src/train.py --encoding onehot --model cnn --epochs 30 --batch-size 32
```

Train a U-Net model with BLOSUM62 encoding:
```bash
python src/train.py --encoding blosum --model unet --epochs 50
```

Train with ESM2 embeddings (requires GPU with sufficient memory):
```bash
python src/train.py --encoding esm2 --model lstm --epochs 30 --batch-size 8
```

### 2. Evaluating a Model

Evaluate on test set:
```bash
python src/evaluate.py --model-dir runs/run_001 --test-file ../rcl-unet/data/Uniprot_Test_Set.csv
```

### 3. Predicting RCL in New Sequences

Annotate sequences in FASTA format:
```bash
python src/predict.py \
    --input my_proteins.fasta \
    --output annotated_proteins.fasta \
    --model-dir runs/run_001
```

Save detailed predictions to CSV:
```bash
python src/predict.py \
    --input my_proteins.fasta \
    --output annotated_proteins.fasta \
    --model-dir runs/run_001 \
    --csv predictions.csv
```

## Configuration

Edit `config.yaml` to modify default settings:
- Data paths
- Model hyperparameters
- Training parameters
- Hardware settings

## Output Formats

### Training Output
Each run creates a directory (`runs/run_XXX/`) containing:
- `best_model.pt` - Best model checkpoint
- `config.json` - Run configuration
- `training_history.json` - Training metrics
- `loss_curves.png` - Training/validation loss plots
- `metric_curves.png` - Validation metrics plots

### Evaluation Output
Evaluation creates a subdirectory with:
- `metrics.json` - Performance metrics
- `predictions.csv` - Detailed per-protein predictions
- `confusion_matrix.png` - Confusion matrix visualization
- `error_histogram.png` - Error distribution

### Prediction Output (FASTA)
```
>ProteinID rcl:350-370 score:0.95
MYLKIVILVTFPLVCFTQDDTPL...
```

## Model Architectures

### CNN
- Multi-layer 1D convolutional network
- Batch normalization and dropout
- Good balance of speed and accuracy

### U-Net
- Encoder-decoder with skip connections
- Attention gates for better localization
- Best for precise RCL boundary detection

### LSTM
- Bidirectional LSTM layers
- Captures long-range dependencies
- Good for variable-length RCLs

## Encoding Schemes

### One-Hot (21-dimensional)
- Fast encoding
- Works well for simple patterns
- Recommended for initial experiments

### BLOSUM62 (20-dimensional)
- Evolutionary information
- Better generalization
- Recommended for production use

### ESM2 (1280-dimensional)
- Protein language model embeddings
- Best performance
- Requires GPU and more memory
- Slower encoding

## Tips for Better Performance

1. **Class Imbalance**: Adjust `loss_weights` in config.yaml if RCL regions are rare
2. **Early Stopping**: Increase `patience` for more training epochs
3. **Learning Rate**: Try different values (0.0001 - 0.001)
4. **Ensemble**: Train multiple models and average predictions
5. **Data Augmentation**: Consider adding more training data

## Troubleshooting

### Out of Memory (GPU)
- Reduce `batch_size`
- Use smaller model or encoding
- For ESM2, reduce `batch_size` in encoding config

### Poor Performance
- Try different model architectures
- Adjust loss weights for class imbalance
- Increase training epochs
- Use BLOSUM or ESM2 encoding

### Slow Training
- Use GPU (`--device cuda`)
- Increase `num_workers` in config
- Use simpler encoding (one-hot or BLOSUM)

## Comparison with Old Implementation

### Improvements
1. **Framework**: PyTorch vs TensorFlow (better flexibility)
2. **Modularity**: Separate data/models/utils
3. **Encodings**: Simplified ESM2 integration
4. **Models**: More architecture options
5. **Inference**: Direct FASTA input/output
6. **Metrics**: Comprehensive evaluation
7. **Logging**: TensorBoard integration

### Migration Notes
- Old TensorFlow models are not compatible
- Need to retrain models with new code
- Data format remains the same (CSV with rcl_start/rcl_end)
- Performance should be similar or better
