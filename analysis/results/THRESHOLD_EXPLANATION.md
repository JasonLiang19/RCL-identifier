# Threshold Parameter Explanation: Why 0.5 is Default

## Quick Answer

**Threshold = 0.5** is the default because:
1. It's the **neutral decision boundary** for binary classification (50/50 split)
2. Models output **calibrated probabilities** via softmax
3. It provides **balanced precision and recall**
4. It's the **standard default** in binary classification tasks

---

## How Threshold Works in Prediction

### Step 1: Model Output (Raw Logits)

The model outputs **raw scores (logits)** for each position in the sequence:

```python
outputs = model(encodings_batch)  # Shape: (batch, 1024, 2)
# Example output for one position:
# outputs[0, 100] = [-2.3, 1.8]  # [non-RCL score, RCL score]
```

These are **unbounded values** (can be any number from -∞ to +∞).

### Step 2: Softmax Conversion to Probabilities

The raw logits are converted to **probabilities** using softmax:

```python
probs = torch.softmax(outputs, dim=-1)  # Shape: (batch, 1024, 2)
# Example for the same position:
# probs[0, 100] = [0.02, 0.98]  # [P(non-RCL), P(RCL)]
```

**Key properties of softmax probabilities:**
- Always sum to 1.0: `P(non-RCL) + P(RCL) = 1.0`
- Range: Each probability is between 0.0 and 1.0
- Interpretation: `P(RCL) = 0.98` means "98% confident this position is RCL"

### Step 3: Thresholding (Decision Making)

The threshold determines which positions are classified as RCL:

```python
# Line 215 in predict.py
rcl_pred = (pred_probs[:seq_length, 1] > threshold).astype(int)
```

**How it works:**
```python
# For each position in the sequence:
for position in range(seq_length):
    rcl_probability = pred_probs[position, 1]  # Probability of being RCL
    
    if rcl_probability > threshold:
        prediction = "RCL"      # Position is RCL
    else:
        prediction = "non-RCL"  # Position is not RCL
```

**Examples with different thresholds:**

| Position | P(RCL) | threshold=0.3 | threshold=0.5 | threshold=0.7 |
|----------|--------|---------------|---------------|---------------|
| 100      | 0.98   | **RCL** ✓     | **RCL** ✓     | **RCL** ✓     |
| 101      | 0.75   | **RCL** ✓     | **RCL** ✓     | **RCL** ✓     |
| 102      | 0.55   | **RCL** ✓     | **RCL** ✓     | non-RCL       |
| 103      | 0.45   | **RCL** ✓     | non-RCL       | non-RCL       |
| 104      | 0.20   | non-RCL       | non-RCL       | non-RCL       |

---

## Why 0.5 is the Default

### 1. Neutral Decision Boundary

Since probabilities sum to 1.0:
- `P(RCL) > 0.5` means `P(RCL) > P(non-RCL)` 
- The model is **more confident** about RCL than non-RCL
- This is the natural decision boundary

**Mathematically:**
```
If P(RCL) + P(non-RCL) = 1.0:
  - P(RCL) > 0.5  →  P(RCL) > P(non-RCL)  →  Predict RCL
  - P(RCL) < 0.5  →  P(RCL) < P(non-RCL)  →  Predict non-RCL
  - P(RCL) = 0.5  →  Equal confidence (tie)
```

### 2. Balanced Precision and Recall

With threshold = 0.5:
- **Precision**: High (few false positives)
- **Recall**: High (few false negatives)
- **F1 Score**: Optimal balance

**Test set results with threshold = 0.5:**

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| onehot_unet | 99.11% | 100.0% | 99.55% |
| blosum_unet | 99.20% | 100.0% | 99.60% |
| esm2_650m_unet | 99.81% | 100.0% | 99.91% |

### 3. Well-Calibrated Models

Our models were trained with:
- Binary cross-entropy loss
- Softmax output layer
- Balanced training data

This produces **well-calibrated probabilities**:
- P(RCL) ≈ 0.9 means ~90% chance of being RCL
- P(RCL) ≈ 0.5 means ~50% chance (uncertain)
- P(RCL) ≈ 0.1 means ~10% chance (likely non-RCL)

With calibrated probabilities, **0.5 is the natural threshold**.

### 4. Standard Practice

In binary classification, **0.5 is the default threshold** because:
- Most common in machine learning literature
- Easy to interpret (more confident than not)
- Works well for balanced problems
- Starting point for threshold tuning

---

## When to Change the Threshold

### Lower Threshold (e.g., 0.3): More Sensitive

**Use when:**
- You want to **catch all possible RCLs** (high recall)
- False positives are acceptable
- Screening/discovery phase

**Effect:**
```python
threshold = 0.3
# Predicts RCL if P(RCL) > 0.3
```

**Trade-off:**
- ✅ Higher recall (catches more true RCLs)
- ✅ Fewer false negatives (missed RCLs)
- ⚠️ Lower precision (more false positives)
- ⚠️ More noisy predictions

**Example results (hypothetical):**
- Recall: 100% → 100%
- Precision: 99% → 85%
- F1: 99.5% → 92%

### Higher Threshold (e.g., 0.7): More Specific

**Use when:**
- You want **high-confidence predictions only**
- False positives are costly
- Validation/confirmation phase

**Effect:**
```python
threshold = 0.7
# Predicts RCL if P(RCL) > 0.7
```

**Trade-off:**
- ✅ Higher precision (fewer false positives)
- ✅ More confident predictions
- ⚠️ Lower recall (may miss some RCLs)
- ⚠️ More conservative

**Example results (hypothetical):**
- Precision: 99% → 99.9%
- Recall: 100% → 95%
- F1: 99.5% → 97.4%

---

## Practical Examples

### Example 1: Screening Unknown Proteins (Lower Threshold)

```bash
python prediction/predict.py novel_proteins.fasta \
    --model esm2_650m_unet \
    --threshold 0.3 \
    --min-consecutive 15
```

**Rationale:**
- Cast a wide net to find all potential RCLs
- Use min-consecutive filter to reduce noise
- Review predictions manually for high-value targets

### Example 2: High-Confidence Annotation (Default)

```bash
python prediction/predict.py database.fasta \
    --model esm2_650m_unet \
    --threshold 0.5
```

**Rationale:**
- Balanced accuracy for database annotation
- Standard threshold for publication
- Proven performance on test set

### Example 3: Conservative Validation (Higher Threshold)

```bash
python prediction/predict.py candidates.fasta \
    --model esm2_650m_unet \
    --threshold 0.8 \
    --min-consecutive 12
```

**Rationale:**
- Only report very high-confidence RCLs
- Minimize false positives for experimental validation
- Worth the cost of missing a few edge cases

---

## Threshold Selection Guide

### Decision Tree

```
Are false positives costly (e.g., experimental validation)?
├─ YES → Use higher threshold (0.7-0.9)
└─ NO → Continue...

Do you need to find all possible RCLs?
├─ YES → Use lower threshold (0.3-0.4)
└─ NO → Use default (0.5)
```

### Recommended Thresholds by Use Case

| Use Case | Threshold | Min-Consecutive | Rationale |
|----------|-----------|-----------------|-----------|
| **Database annotation** | 0.5 | 12 | Balanced, proven performance |
| **Discovery/screening** | 0.3-0.4 | 15-20 | High recall, filter noise |
| **Experimental validation** | 0.7-0.8 | 12 | High confidence only |
| **Publication figures** | 0.5 | 12 | Standard, reproducible |
| **High-throughput filtering** | 0.6 | 15 | Slight bias toward precision |

---

## Technical Details: Model Training

### Why Our Models Work Well at 0.5

During training, the model was optimized with:

```python
# Binary cross-entropy loss
loss = -[y * log(p) + (1-y) * log(1-p)]

# Where:
# y = true label (0 or 1)
# p = predicted probability (from softmax)
```

This loss function:
1. **Penalizes confident wrong predictions** more than uncertain ones
2. **Rewards calibrated probabilities** (p ≈ true frequency)
3. **Optimizes for threshold = 0.5** implicitly

The model learns to output:
- P(RCL) ≈ 1.0 for clear RCL positions
- P(RCL) ≈ 0.0 for clear non-RCL positions
- P(RCL) ≈ 0.5 only when truly uncertain

---

## Visualizing Threshold Effect

Imagine a histogram of predicted probabilities for all positions:

```
Threshold = 0.3:  |----[RCL]-----------|--[non-RCL]--|
                  0.0  0.3            0.7            1.0
                       ↑ threshold

Threshold = 0.5:  |-------[RCL]--------|----[non-RCL]---|
                  0.0       0.5                         1.0
                            ↑ threshold (default)

Threshold = 0.7:  |----------[RCL]-----|--[non-RCL]---|
                  0.0              0.7              1.0
                                   ↑ threshold
```

**Typical probability distribution for our models:**
```
Non-RCL positions: P(RCL) ∈ [0.0, 0.2]  ← Most predictions here
Uncertain:         P(RCL) ∈ [0.2, 0.8]  ← Very few predictions
RCL positions:     P(RCL) ∈ [0.8, 1.0]  ← Most true RCLs here
```

With this distribution, **threshold = 0.5** cleanly separates the two classes.

---

## Summary

**Why threshold = 0.5?**

1. ✅ **Natural boundary**: More confident about RCL than non-RCL
2. ✅ **Balanced performance**: Optimal precision-recall trade-off
3. ✅ **Calibrated probabilities**: Models are well-trained
4. ✅ **Standard practice**: Default in binary classification
5. ✅ **Empirically validated**: 99%+ F1 on test set

**When to change:**
- **Lower (0.3-0.4)**: Maximize recall for discovery
- **Higher (0.7-0.9)**: Maximize precision for validation
- **Keep 0.5**: For standard annotation and publication

**In practice:** Most users should stick with **0.5** (default) unless they have a specific reason to prioritize either recall or precision over the other.
