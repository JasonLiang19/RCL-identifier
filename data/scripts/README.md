# Data Scripts

Utility scripts for data validation and cleaning.

## data_cleaning.py

Validates RCL annotation CSV files and reports integrity issues.

### Features

- ✅ Validates required columns (id, Sequence)
- ✅ Checks RCL annotation fields (rcl_start, rcl_end, rcl_seq)
- ✅ Detects invalid amino acids in sequences
- ✅ Validates sequence lengths
- ✅ Checks RCL coordinate consistency
- ✅ Verifies RCL sequence matches coordinates
- ✅ Reports row numbers for all issues
- ✅ Generates detailed validation reports

### Usage

```bash
# Check single file
python data/scripts/data_cleaning.py --input data/raw/Alphafold_RCL_annotations.csv

# Check multiple files
python data/scripts/data_cleaning.py --input data/raw/*.csv

# With custom max sequence length
python data/scripts/data_cleaning.py --input data/raw/file.csv --max-length 2048

# Save validation reports
python data/scripts/data_cleaning.py --input data/raw/*.csv --output data/validation_reports/
```

### Output Example

```
Validating: data/raw/Alphafold_RCL_annotations.csv
================================================================================
Total rows: 2000

================================================================================
VALIDATION SUMMARY
================================================================================

Statistics:
  Missing ID:                    0
  Missing sequence:              2
  Empty sequence:                1
  Sequence too long (>1024):     15
  Invalid amino acids:           0
  Missing RCL annotation:        503
  Non-numeric indices:           5
  Invalid start index:           0
  End before start:              2
  Start beyond sequence:         3
  End beyond sequence:           8
  Mismatched RCL sequence:       12
  Short RCL (<10 residues):      45
  Long RCL (>50 residues):       3

================================================================================
ERRORS: 33
================================================================================
Row     3 | Field: Sequence       | Missing sequence | Value: None
Row    15 | Field: rcl_start/end  | Non-numeric RCL indices | Value: start=no result, end=no end
Row    27 | Field: rcl_end        | End index less than start index | Value: start=350, end=320
Row    45 | Field: rcl_end        | End index beyond sequence length | Value: end=1500, seq_len=450
Row    67 | Field: rcl_seq        | RCL sequence does not match coordinates | Value: expected=AAAA, got=TTTT
...

================================================================================
WARNINGS: 551
================================================================================
Row    10 | Field: rcl_start/end  | Missing or invalid RCL annotation | Value: start=no result, end=nan
Row    22 | Field: Sequence       | Sequence too long (>1024) | Value: 1250
Row    38 | Field: rcl_length     | RCL very short (8 residues) | Value: start=350, end=357
...

================================================================================
❌ VALIDATION FAILED: 33 errors found
================================================================================
```

### Validation Checks

**Errors** (will cause data loading to fail):
- Missing ID or sequence
- Empty sequences
- Invalid amino acid characters
- Non-numeric RCL indices
- End index before start index
- Indices beyond sequence length
- RCL sequence mismatch with coordinates

**Warnings** (sequences will be skipped but not cause errors):
- Missing RCL annotations
- Sequences longer than max_length
- Very short RCL regions (<10 residues)
- Very long RCL regions (>50 residues)

### Integration with Training

The validation script checks for the same issues that would cause:
1. **Loading errors**: Missing data, invalid formats
2. **Skipped sequences**: Missing annotations (as per updated data_loader.py)
3. **Training issues**: Invalid coordinates, mismatched sequences

Run this before training to identify and fix data issues proactively!
