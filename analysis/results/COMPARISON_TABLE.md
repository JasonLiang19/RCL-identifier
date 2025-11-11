# Model Comparison: All 6 Trained Models

## Overview
This document compares all 6 trained models across both validation and test sets.

**Training Data**: 3,431 sequences (1,384 serpins + 2,048 non-serpins)
- Train: 2,744 sequences (80%)
- Validation: 687 sequences (20%)

**Test Data**: 78 sequences from Uniprot_Test_Set.csv (independent test set)

---

## Performance Comparison

### Validation Set Performance (687 sequences)

| Encoding  | Model | Accuracy | Precision | Recall | F1 Score | F1-Macro | MCC    | AUC    |
|-----------|-------|----------|-----------|--------|----------|----------|--------|--------|
| onehot    | CNN   | 0.9969   | 0.9135    | 0.9796 | 0.9454   | 0.9696   | 0.9439 | 0.9990 |
| onehot    | UNET  | 0.9976   | 0.9429    | 0.9743 | 0.9584   | 0.9775   | 0.9574 | 0.9992 |
| blosum    | CNN   | 0.9975   | 0.9375    | 0.9786 | 0.9576   | 0.9771   | 0.9566 | 0.9993 |
| blosum    | UNET  | 0.9975   | 0.9375    | 0.9759 | 0.9563   | 0.9765   | 0.9553 | 0.9991 |
| esm2_650m | CNN   | 0.9980   | 0.9580    | 0.9703 | 0.9641   | 0.9811   | 0.9632 | 0.9993 |
| esm2_650m | UNET  | 0.9988   | 0.9812    | 0.9693 | **0.9752** | 0.9871   | 0.9748 | 0.9996 |

**Best Model (Validation)**: ESM2_650M + U-Net (F1 = 0.9752)

---

### Test Set Performance (78 sequences)

| Encoding  | Model | Accuracy | Precision | Recall | F1 Score | F1-Macro | MCC    | AUC    |
|-----------|-------|----------|-----------|--------|----------|----------|--------|--------|
| onehot    | CNN   | 0.3997   | 0.0364    | 0.8528 | 0.0699   | 0.3134   | 0.0794 | 0.8617 |
| onehot    | UNET  | 0.8819   | 0.1831    | 1.0000 | **0.3095** | 0.6225   | 0.4011 | 1.0000 |
| blosum    | CNN   | 0.3990   | 0.0370    | 0.8684 | 0.0710   | 0.3134   | 0.0842 | 0.8907 |
| blosum    | UNET  | 0.4275   | 0.0442    | 1.0000 | 0.0846   | 0.3341   | 0.1349 | 0.9993 |
| esm2_650m | CNN   | 0.4055   | 0.0426    | 1.0000 | 0.0817   | 0.3211   | 0.1288 | 1.0000 |
| esm2_650m | UNET  | 0.4171   | 0.0434    | 1.0000 | 0.0832   | 0.3280   | 0.1320 | 1.0000 |

**Best Model (Test)**: OneHot + U-Net (F1 = 0.3095)

---

## Key Observations

### 1. **Validation vs Test Performance Gap**
- **Validation**: All models achieve F1 scores between 0.94-0.98 (excellent)
- **Test**: F1 scores drop to 0.07-0.31 (poor to moderate)
- **Reason**: High recall (85-100%) but very low precision (3.6-18.3%)

### 2. **Class Imbalance Effect**
The test set metrics are calculated per-residue:
- Total positions: 78 sequences × 1024 max length ≈ 80,000 positions
- True RCL positions: ~1,500-2,000 (approximately 2-3% of total)
- Models correctly identify most/all RCL residues (high recall)
- But also predict many non-RCL residues as RCL (low precision)

### 3. **Model Rankings Change**
- **Validation best**: ESM2_650M + U-Net (F1 = 0.9752)
- **Test best**: OneHot + U-Net (F1 = 0.3095)
- The more complex ESM2 model may be overfitting to the training distribution

### 4. **Perfect Recall on Test**
Most U-Net models achieve 100% recall on test set, meaning they capture all true RCL residues but with many false positives.

---

## Analysis

### Why the Performance Gap?

1. **Domain Shift**: Training data from AlphaFold annotations, test data from Uniprot
2. **Class Imbalance**: Per-residue metrics are dominated by the majority (non-RCL) class
3. **Different Sequence Characteristics**: Test sequences may have different properties
4. **Threshold Effects**: Default 0.5 threshold may not be optimal for this task

### Recommendations

1. **Threshold Optimization**: Test higher thresholds (0.6, 0.7, 0.8) to reduce false positives
2. **Post-Processing**: Filter predictions by:
   - Minimum consecutive residues (RCLs are typically 15-30 residues)
   - Length constraints
   - Structural context
3. **Alternative Metrics**: Consider sequence-level metrics:
   - Does the sequence have an RCL? (binary classification)
   - IoU (Intersection over Union) for predicted vs true RCL
4. **Re-training**: Consider including Uniprot data in training set

---

## Training Information

| Encoding  | Model | Best Epoch | Total Epochs | Early Stop |
|-----------|-------|------------|--------------|------------|
| onehot    | CNN   | 6          | 14           | Yes        |
| onehot    | UNET  | 8          | 16           | Yes        |
| blosum    | CNN   | 9          | 17           | Yes        |
| blosum    | UNET  | 9          | 17           | Yes        |
| esm2_650m | CNN   | 4          | 12           | Yes        |
| esm2_650m | UNET  | 6          | 9            | Yes        |

All models used early stopping with patience=7 epochs.

---

## Files Generated

- **Training**: 6 directories in `analysis/results/`, each containing:
  - `best_model.pt` - Model checkpoint at best validation F1
  - `final_model.pt` - Model checkpoint at end of training
  - `training_history.json` - Loss and metrics per epoch
  - `config.json` - Model configuration

- **Evaluation**:
  - `test_set_evaluation.csv` - Test metrics in CSV format
  - `test_set_evaluation.txt` - Formatted test results
  - `COMPARISON_TABLE.md` - This file

---

## Conclusion

While validation performance is excellent (F1 ~ 0.95-0.98), **test set performance reveals significant generalization challenges** (F1 ~ 0.07-0.31). The models achieve high recall but suffer from low precision, suggesting they are **over-predicting RCL regions** on independent test data.

For production use, consider:
- Using **OneHot + U-Net** (best on test set)
- Applying **threshold optimization** (try 0.7-0.8 instead of 0.5)
- Adding **post-processing filters** to reduce false positives
- Evaluating on **sequence-level metrics** rather than per-residue

The current models are functional but will require additional refinement for reliable production deployment.
