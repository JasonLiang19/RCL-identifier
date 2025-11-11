# Filename Standardization - November 10, 2025

## Changes Made

All data filenames have been standardized to use underscores instead of spaces for better compatibility with command-line tools and scripts.

### Files Renamed

1. **Alphafold RCL annotations.csv** → **Alphafold_RCL_annotations.csv**
   - Location: `data/raw/`
   - Contains: 1,384 cleaned serpin sequences with valid RCL annotations
   - Backup: `old_RCL_annotation_file.csv` (1,962 rows - original with invalid data)

2. **Uniprot Test Set.csv** → **Uniprot_Test_Set.csv**
   - Location: `data/raw/`
   - Already had underscores, verified in config

### Files Updated

The following files were updated to use the new filenames:

1. **config.yaml**
   - `train_serpin`: "data/raw/Alphafold_RCL_annotations.csv"
   - `test_serpin`: "data/raw/Uniprot_Test_Set.csv"

2. **examples.py**
   - Updated reference to Alphafold_RCL_annotations.csv

3. **QUICK_REFERENCE.md**
   - Updated training and test file paths

4. **data/scripts/README.md**
   - Updated example usage commands

5. **data/raw/FILTERING_SUMMARY.txt**
   - Updated current filename reference

### Verification

All data loading has been tested and verified:
- ✅ Serpin sequences: 1,383 loaded from Alphafold_RCL_annotations.csv
- ✅ Non-serpin sequences: 2,048 loaded from non_serpin_train.csv
- ✅ Total training sequences: 3,431
- ✅ Config file correctly references all data files

### Benefits

1. **Command-line friendly**: No need for escaping spaces in bash commands
2. **Cross-platform**: Works consistently on Windows, Linux, and macOS
3. **Python-friendly**: Easier to handle in scripts without quotes
4. **Git-friendly**: Better compatibility with version control systems
5. **Consistency**: All data files now follow the same naming convention

### Standard Naming Convention

Going forward, all data files should follow this pattern:
- Use underscores (`_`) instead of spaces
- Use capital letters for proper nouns (e.g., `Alphafold`, `Uniprot`)
- Use descriptive names (e.g., `RCL_annotations`, `Test_Set`)

### Example Usage

```bash
# No escaping needed
python src/train.py --encoding onehot --model cnn

# Data validation
python data/scripts/data_cleaning.py --input data/raw/Alphafold_RCL_annotations.csv

# Preprocessing
python src/precompute_embeddings.py --encoding esm2_650m
```

All scripts and documentation have been updated to reflect these changes.
