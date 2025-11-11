# RCL Prediction

This directory contains tools for predicting Reactive Center Loop (RCL) locations in protein sequences using trained models.

## Quick Start

**Note:** Run these commands from the `prediction/` directory.

```bash
# Basic prediction - outputs TSV with RCL locations
python predict.py input/Dmel_6.54.fasta --model onehot_unet
python predict.py input/Dmel_6.54.fasta --model blosum_unet
python predict.py input/Dmel_6.54.fasta --model esm2_650m_unet

# With additional FASTA outputs
python predict.py input/Dmel_6.54.fasta --model esm2_650m_unet -seq_fasta -RCL_fasta
```

## Available Models

Three high-performance U-Net models are available:

| Model | Seq-Level IoU | Res-Level F1 | Description |
|-------|---------------|--------------|-------------|
| `onehot_unet` | 0.9943 | 0.9955 | One-hot encoding + U-Net (fast, lightweight) |
| `blosum_unet` | 0.9941 | 0.9960 | BLOSUM62 encoding + U-Net (fast, accurate) |
| `esm2_650m_unet` | 0.9982 | 0.9991 | ESM2-650M encoding + U-Net (best performance) |

**All models achieve 100% binary classification** (correctly identify RCL-containing sequences) and **97-99% exact match** on test set.

**Recommendation:**
- **Best performance**: `esm2_650m_unet` (99.8% IoU, 98.7% exact match)
- **Best lightweight**: `onehot_unet` or `blosum_unet` (99.4% IoU, 97.4% exact match, much faster)

## Usage

### Basic Usage

```bash
python predict.py <input.fasta> --model <model_name>
```

**Example:**
```bash
# Use OneHot model (fastest, very accurate)
python predict.py input/Dmel_6.54.fasta --model onehot_unet

# Use BLOSUM model (fast, very accurate)
python predict.py input/Dmel_6.54.fasta --model blosum_unet

# Use ESM2 model (best accuracy, slower)
python predict.py input/Dmel_6.54.fasta --model esm2_650m_unet
```

This will create: `prediction/output/Dmel_6.54_onehot_RCL_predictions_thr0p50.tsv` (or `blosum` or `esm2_650m` depending on model)

### Command Line Options

```
positional arguments:
  input                 Input FASTA file with protein sequences

required arguments:
  --model {onehot_unet,blosum_unet,esm2_650m_unet}
                        Model to use for prediction

optional arguments:
  --output-dir DIR      Output directory (default: prediction/output/)
  --min-rcl-length N    Minimum amino acids to identify as RCL (default: 7)
  --min-consecutive N   Minimum consecutive residues for valid RCL (default: 12)
  --threshold T         Probability threshold for prediction (default: 0.5)
  -seq_fasta            Generate FASTA with sequences having RCL
  -RCL_fasta            Generate FASTA with only RCL sequences
  --batch-size N        Batch size for prediction (default: 8)
  --chunk-size N        Process in chunks to save memory (default: 10000)
  --device {cuda,cpu}   Device to use (default: auto-detect)
```

## Output Formats

### Default Output: TSV File

Always generated. Contains RCL locations for all sequences with identified RCLs.

**Format:** `<basename>_<encoding>_RCL_predictions_thr<threshold>.tsv`

Where:
- `<basename>`: Input filename without extension
- `<encoding>`: Encoding type (onehot, blosum, or esm2_650m)
- `<threshold>`: Threshold value (e.g., thr0p50 for 0.5)

**Example:** `proteins_esm2_650m_RCL_predictions_thr0p50.tsv`

**Columns:**
- `Accession`: Protein accession (text immediately after `>` in input FASTA)
- `RCL_start_position`: Start position of RCL (1-indexed)
- `RCL_end_position`: End position of RCL (1-indexed, exclusive)

**Example:**
```
Accession       RCL_start_position      RCL_end_position
FBpp0070000     342                     360
FBpp0300206     342                     360
```

### Optional Output 1: Sequences with RCL (`-seq_fasta`)

FASTA file containing all sequences that have an identified RCL.

**Format:** `<basename>_<encoding>_with_RCL_thr<threshold>.fasta`

**Example:** `proteins_blosum_with_RCL_thr0p50.fasta`

**Header format:** `>Accession`

**Example:**
```fasta
>FBpp0070000
MTRYKQTEFTEDDSSSIGGIQLNEATGHTGMQIRYHTARATWNWRSRNKTEKWLLITTFV
MAITIFTLLIVLFTDGGSSDATKHVLHVQPHQKDCPSGNELPCLNKHCIFASSEILKSI
...
```

### Optional Output 2: RCL Sequences Only (`-RCL_fasta`)

FASTA file containing only the RCL sequences extracted from proteins.

**Format:** `<basename>_<encoding>_RCL_only_thr<threshold>.fasta`

**Example:** `proteins_onehot_RCL_only_thr0p50.fasta`

**Header format:** `>Accession RCL {Start} - {End}`

**Example:**
```fasta
>FBpp0070000 RCL 342 - 360
NFNCLFGWAIGEDDKNSS
>FBpp0300206 RCL 342 - 360
NFNCLFGWAIGEDDKNSS
```

## Examples

### Example 1: Basic Prediction with OneHot

```bash
python predict.py input/my_proteins.fasta --model onehot_unet
```

**Output:**
- `prediction/output/my_proteins_onehot_RCL_predictions_thr0p50.tsv`

### Example 2: Complete Output with ESM2

```bash
python predict.py input/proteome.fasta --model esm2_650m_unet -seq_fasta -RCL_fasta
```

**Output:**
- `prediction/output/proteome_esm2_650m_RCL_predictions_thr0p50.tsv` (RCL locations)
- `prediction/output/proteome_esm2_650m_with_RCL_thr0p50.fasta` (full sequences with RCL)
- `prediction/output/proteome_esm2_650m_RCL_only_thr0p50.fasta` (extracted RCL sequences)

### Example 3: Custom Output Directory

```bash
python predict.py input/proteins.fasta --model blosum_unet \
    --output-dir results/batch1/ \
    -seq_fasta -RCL_fasta
```

**Output:**
- `results/batch1/proteins_blosum_RCL_predictions_thr0p50.tsv`
- `results/batch1/proteins_blosum_with_RCL_thr0p50.fasta`
- `results/batch1/proteins_blosum_RCL_only_thr0p50.fasta`

### Example 4: Custom Minimum Consecutive Residues (Post-Processing Filter)

```bash
python predict.py input/proteins.fasta --model onehot_unet \
    --min-consecutive 15
```

Only reports RCLs with ≥15 consecutive predicted residues (reduces false positives).

### Example 5: Lower Threshold for More Sensitivity

```bash
python predict.py input/proteins.fasta --model blosum_unet \
    --threshold 0.3
```

Uses lower probability threshold (more sensitive, may have more false positives).

## Input Format

Input must be a standard FASTA file with protein sequences.

**Requirements:**
- Sequences must be ≤1024 amino acids
- Standard amino acid alphabet
- One or more sequences

**Example:**
```fasta
>UniProtID|Description here
MKTLLLTLVVVTIVFPFPLNHAFTQPSSETPIVKQFNNQYKVEGEQWHHFLFDDDPQNQV
SFNDPNYRGFSSVYLPGTQPVVRFETKFLEFLEDSLFKLQPESEKLYSLPPEGHLQNQKS
```

**Accession Parsing:**
- Accession is extracted as the first word after `>`
- In the above example: `UniProtID|Description` → Accession = `UniProtID|Description`
- If header is `>sp|P12345|SERPA_HUMAN`, Accession = `sp|P12345|SERPA_HUMAN`
- If you want just the ID, ensure input headers are formatted appropriately

## Model Performance

All three models were trained on 3,431 sequences and evaluated on an independent test set of 78 sequences.

### Sequence-Level Performance (Recommended Metric)

| Model | Binary F1 | Mean IoU | Exact Match | Speed |
|-------|-----------|----------|-------------|-------|
| `onehot_unet` | 1.0000 | 0.9943 | 97.4% | Fast |
| `blosum_unet` | 1.0000 | 0.9941 | 97.4% | Fast |
| `esm2_650m_unet` | 1.0000 | 0.9982 | 98.7% | Moderate |

- **Binary F1**: Accuracy of identifying which sequences contain RCL (100% for all models)
- **Mean IoU**: Intersection over Union between predicted and true RCL regions
- **Exact Match**: % of sequences with RCL position within ±2 residues

### Residue-Level Performance (Per-Position Classification)

| Model | Accuracy | Precision | Recall | F1 | MCC | AUC |
|-------|----------|-----------|--------|-----|-----|-----|
| `onehot_unet` | 99.94% | 0.991 | 1.000 | 0.996 | 0.995 | 1.000 |
| `blosum_unet` | 99.95% | 0.992 | 1.000 | 0.996 | 0.996 | 1.000 |
| `esm2_650m_unet` | 99.99% | 0.998 | 1.000 | 0.999 | 0.999 | 1.000 |

**All models show excellent performance with >99% accuracy and F1 scores.**

## Minimum Consecutive Residues (Post-Processing Filter)

By default (`--min-consecutive 12`), predicted RCL regions must have ≥12 consecutive residues to be reported.

This post-processing filter helps reduce false positives from scattered predictions while maintaining high recall.

**To adjust:**
```bash
# More strict (longer consecutive regions required)
python predict.py input.fasta --model onehot_unet --min-consecutive 20

# Less strict (allow shorter regions)
python predict.py input.fasta --model blosum_unet --min-consecutive 8

# No filtering (minimum 1 residue)
python predict.py input.fasta --model esm2_650m_unet --min-consecutive 1
```

**Note:** This is separate from `--min-rcl-length` which filters the final reported RCLs.

## Minimum RCL Length

After post-processing, sequences must have an identified RCL ≥7 amino acids (by default) to be reported.

**To adjust:**
```bash
python predict.py input.fasta --model blosum_unet --min-rcl-length 10
```

## Prediction Threshold

The threshold determines when a position is classified as RCL based on the model's probability output.

### How It Works

1. **Model outputs probabilities** for each position: `[P(non-RCL), P(RCL)]`
2. **Threshold applies to RCL probability**: If `P(RCL) > threshold`, predict RCL
3. **Default = 0.5** means: predict RCL when model is more confident about RCL than non-RCL

### Why 0.5 is Default

- **Neutral boundary**: P(RCL) > 0.5 means model favors RCL over non-RCL
- **Balanced performance**: Our models achieve 99%+ precision and recall at 0.5
- **Well-calibrated**: Models output accurate probabilities
- **Standard practice**: Default in binary classification

### When to Change

**Lower threshold (0.3-0.4) for higher sensitivity:**
```bash
# Find more potential RCLs (higher recall, more false positives)
python predict.py input.fasta --model blosum_unet --threshold 0.3
```
Use for: Discovery, screening, maximizing recall

**Higher threshold (0.7-0.8) for higher specificity:**
```bash
# Only high-confidence RCLs (higher precision, may miss some)
python predict.py input.fasta --model blosum_unet --threshold 0.7
```
Use for: Experimental validation, conservative annotation

**Keep default (0.5) for standard annotation:**
```bash
# Balanced, proven performance
python predict.py input.fasta --model blosum_unet --threshold 0.5
```

### Example Impact

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3 | ~85-95% | ~100% | Discovery/screening |
| 0.5 (default) | ~99% | ~100% | Standard annotation |
| 0.7 | ~99.9% | ~95-100% | Validation/publication |

**Recommendation:** Start with 0.5 (default). Only change if you have specific recall or precision requirements.

## Performance Considerations

### Speed

- **OneHot model**: ~150-250 sequences/second (very fast, lightweight)
- **BLOSUM model**: ~100-200 sequences/second (very fast)
- **ESM2 model**: ~5-10 sequences/second (slower due to transformer)

For large datasets (>10,000 sequences), consider using `onehot_unet` or `blosum_unet` for faster processing.

### Memory

- **OneHot model**: Very low memory (~1.5 GB GPU)
- **BLOSUM model**: Low memory (~2 GB GPU)
- **ESM2 model**: Higher memory (~8 GB GPU for batch_size=8)

If running out of GPU memory with ESM2, reduce batch size:
```bash
python predict.py input.fasta --model esm2_650m_unet --batch-size 4
```

For very large datasets, use chunked processing (automatic for >10k sequences):
```bash
python predict.py large_proteome.fasta --model blosum_unet --chunk-size 5000
```

### GPU vs CPU

Models automatically use GPU if available. To force CPU:
```bash
python predict.py input.fasta --model blosum_unet --device cpu
```

**Note:** ESM2 model on CPU will be very slow (recommend GPU).

For fastest CPU performance, use `onehot_unet` or `blosum_unet`:
```bash
python predict.py input.fasta --model onehot_unet --device cpu
```

## Troubleshooting

### Error: "Model directory not found"

Make sure you've run the training experiments:
```bash
cd ..
bash analysis/run_experiments.sh
```

The models should be in `analysis/results/onehot_unet/`, `analysis/results/blosum_unet/`, and `analysis/results/esm2_650m_unet/`.

### Error: "No valid sequences to process"

All sequences exceed max length (1024 aa). Try:
1. Filter input to shorter sequences
2. Split long sequences into domains

### Out of Memory

For ESM2 model:
```bash
python predict.py input.fasta --model esm2_650m_unet --batch-size 4
```

Or use OneHot or BLOSUM model which require less memory:
```bash
python predict.py input.fasta --model onehot_unet
# or
python predict.py input.fasta --model blosum_unet
```

### No RCLs Found

Try lowering the threshold:
```bash
python predict.py input.fasta --model blosum_unet --threshold 0.3 --min-rcl-length 5
```

## Directory Structure

```
prediction/
├── predict.py              # Main prediction script
├── README.md               # This file
├── input/                  # Place input FASTA files here
│   └── Dmel_6.54.fasta     # Example input file
└── output/                 # Prediction results saved here
    ├── *_RCL_predictions.tsv
    ├── *_with_RCL.fasta
    └── *_RCL_only.fasta
```

## Citation

If you use these RCL prediction models, please cite the training data sources and model architecture.

## Support

For issues or questions, please check:
1. This README
2. Main project documentation in `../README.md`
3. Example outputs in `output/` directory
