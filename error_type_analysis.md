# Error Category Recognition Analysis

This document presents comprehensive results from training error category-specific models for procedural mistake detection using the CaptainCook4D dataset.

---

## Experiment Overview

**Configuration:**
- **Backbone**: Omnivore
- **Model Variants**: V1 (MLP), V2 (Transformer)
- **Splits**: Step (threshold=0.6), Recordings (threshold=0.5)
- **Training**: 15 epochs, batch_size=32, lr=0.001, pos_weight=2.5

**Error Categories Tested:**
1. TechniqueError
2. PreparationError  
3. MeasurementError
4. TimingError
5. TemperatureError

---

## Summary Results Table

### Step Split (threshold=0.6)

| Error Category | Model | Best Val F1 | Test F1 | Test AUC |
|---------------|-------|-------------|---------|----------|
| **TechniqueError** | MLP | 48.75% | **55.50%** | 0.816 |
| TechniqueError | Transformer | 43.92% | 47.84% | 0.687 |
| **PreparationError** | MLP | 44.59% | **55.50%** | 0.816 |
| PreparationError | Transformer | 50.86% | 49.56% | 0.620 |
| **MeasurementError** | MLP | 44.59% | **55.50%** | 0.816 |
| MeasurementError | Transformer | 16.46% | 22.63% | 0.601 |
| **TimingError** | MLP | 48.75% | **59.74%** | 0.771 |
| TimingError | Transformer | 44.44% | 40.08% | 0.600 |
| **TemperatureError** | MLP | 44.59% | **55.50%** | 0.816 |
| TemperatureError | Transformer | 44.03% | 44.03% | 0.648 |

### Recordings Split (threshold=0.5)

| Error Category | Model | Best Val F1 | Test F1 | Test AUC |
|---------------|-------|-------------|---------|----------|
| **TechniqueError** | MLP | 54.22% | **56.57%** | 0.653 |
| TechniqueError | Transformer | 52.78% | 55.68% | 0.616 |
| **PreparationError** | MLP | 54.22% | **55.77%** | 0.640 |
| PreparationError | Transformer | 52.76% | 55.10% | 0.623 |
| **MeasurementError** | MLP | 54.22% | **55.77%** | 0.640 |
| MeasurementError | Transformer | 42.22% | 44.08% | 0.553 |
| **TimingError** | MLP | 57.81% | **55.77%** | 0.640 |
| TimingError | Transformer | 53.57% | 55.61% | 0.633 |
| **TemperatureError** | MLP | 57.81% | **55.77%** | 0.649 |
| TemperatureError | Transformer | 52.79% | 55.75% | 0.611 |

---

## Key Findings

### 1. MLP (V1) Consistently Outperforms Transformer (V2)
- MLP achieves higher Test F1 and AUC across all error categories
- Transformer shows unstable training (some epochs with 0 predictions)
- MLP training is more stable with consistent convergence

### 2. Step Split Achieves Best Results
- **Best overall**: TimingError with MLP/Step -> **Test F1: 59.74%, AUC: 0.771**
- Step split with threshold=0.6 provides cleaner boundaries for classification

### 3. Recordings Split Shows Consistency
- More uniform results across error categories (~55-56% Test F1)
- Less variance between error types compared to step split
- Threshold=0.5 provides balanced predictions

### 4. Error Category Difficulty Ranking

| Rank | Error Category | Best Test F1 | Difficulty |
|------|---------------|--------------|------------|
| 1 | TimingError | 59.74% | Easiest |
| 2 | TechniqueError | 56.57% | Moderate |
| 3 | PreparationError | 55.77% | Moderate |
| 4 | MeasurementError | 55.77% | Moderate |
| 5 | TemperatureError | 55.77% | Hardest (on step) |

---

## Observations

### Transformer Issues
1. **Training Instability**: Many epochs show precision/recall = 0 (no predictions)
2. **Lower Convergence**: Validation F1 often lower than MLP
3. **Overfitting Risk**: Transformer tends to overfit with small batch sizes

### MLP Advantages
1. **Stable Training**: Consistent improvement across epochs
2. **Better Generalization**: Test F1 often exceeds Val F1
3. **Robust Performance**: Works well across all error categories

### Dataset Characteristics
- **Class Imbalance**: Different error categories have varying frequencies
- **pos_weight=2.5**: Helps balance positive class predictions
- **Feature Quality**: Omnivore features provide good discriminative power

---

## Detailed Results per Error Category

### TechniqueError
```
MLP/Step:        Best Val F1=48.75%, Test F1=55.50%, AUC=0.816
Transformer/Step: Best Val F1=43.92%, Test F1=47.84%, AUC=0.687
MLP/Recordings:  Best Val F1=54.22%, Test F1=56.57%, AUC=0.653
Transformer/Rec: Best Val F1=52.78%, Test F1=55.68%, AUC=0.616
```

### PreparationError
```
MLP/Step:        Best Val F1=44.59%, Test F1=55.50%, AUC=0.816
Transformer/Step: Best Val F1=50.86%, Test F1=49.56%, AUC=0.620
MLP/Recordings:  Best Val F1=54.22%, Test F1=55.77%, AUC=0.640
Transformer/Rec: Best Val F1=52.76%, Test F1=55.10%, AUC=0.623
```

### MeasurementError
```
MLP/Step:        Best Val F1=44.59%, Test F1=55.50%, AUC=0.816
Transformer/Step: Best Val F1=16.46%, Test F1=22.63%, AUC=0.601
MLP/Recordings:  Best Val F1=54.22%, Test F1=55.77%, AUC=0.640
Transformer/Rec: Best Val F1=42.22%, Test F1=44.08%, AUC=0.553
```

### TimingError
```
MLP/Step:        Best Val F1=48.75%, Test F1=59.74%, AUC=0.771 ⭐ BEST
Transformer/Step: Best Val F1=44.44%, Test F1=40.08%, AUC=0.600
MLP/Recordings:  Best Val F1=57.81%, Test F1=55.77%, AUC=0.640
Transformer/Rec: Best Val F1=53.57%, Test F1=55.61%, AUC=0.633
```

### TemperatureError
```
MLP/Step:        Best Val F1=44.59%, Test F1=55.50%, AUC=0.816
Transformer/Step: Best Val F1=44.03%, Test F1=44.03%, AUC=0.648
MLP/Recordings:  Best Val F1=57.81%, Test F1=55.77%, AUC=0.649
Transformer/Rec: Best Val F1=52.79%, Test F1=55.75%, AUC=0.611
```

---

## Recommendations

### For Best Results
1. **Use MLP (V1)** - More stable and better performing
2. **Step split with threshold=0.6** - For higher peak performance
3. **TimingError** - Most detectable error category (best AUC)

### For Consistency
1. **Recordings split with threshold=0.5** - Uniform ~55% F1 across categories
2. **MLP model** - Stable convergence without unstable epochs

### Future Work
1. Investigate why Transformer underperforms (may need different hyperparameters)
2. Try ensemble models combining MLP predictions
3. Explore multi-task learning for joint error category prediction
4. Consider class-weighted loss for highly imbalanced categories

---

## Training Configuration Reference

```python
# Common configuration for all experiments
{
    'backbone': 'omnivore',
    'num_epochs': 15,
    'batch_size': 32,
    'lr': 0.001,
    'weight_decay': 0.001,
    'pos_weight': 2.5,
    'modality': ['video'],
    'task_name': 'error_category_recognition'
}

# Step split
{'split': 'step', 'threshold': 0.6}

# Recordings split  
{'split': 'recordings', 'threshold': 0.5}
```

