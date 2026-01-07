# LSTM Baseline v2 - Training Results Summary

## 📊 Results (Updated: January 6, 2026)

This document summarizes the LSTM baseline results with **corrected thresholds** per CaptainCook4D paper recommendations.

---

## Training Configuration

- **Epochs**: 15 for Omnivore configs, 30 for SlowFast configs
- **Batch Size**: 32
- **pos_weight**: 2.5
- **Evaluation Threshold**: 
  - Step split: 0.6 ✅
  - Recordings split: 0.5 ✅ (per paper recommendation)

---

## Final Results

### Step Level Metrics Summary (Validation Set - Final Epoch)

| Config | Backbone | Split | Threshold | Epochs | Precision | Recall | F1 | AUC |
|--------|----------|-------|-----------|--------|-----------|--------|-------|-----|
| 1 | Omnivore | step | 0.6 | 15 | 48.22% | 53.80% | **50.85%** | 68.15% |
| 2 | Omnivore | recordings | 0.5 | 15 | **44.62%** | 75.07% | **55.98%** | **67.46%** |
| 3 | SlowFast | step | 0.6 | 30 | 36.59% | **83.74%** | 50.93% | 58.21% |
| 4 | SlowFast | recordings | 0.5 | 30 | 38.02% | 85.11% | 52.56% | 57.93% |

### Test Set Metrics (Final Epoch)

| Config | Backbone | Split | Threshold | Precision | Recall | F1 | AUC |
|--------|----------|-------|-----------|-----------|--------|-------|-----|
| 1 | Omnivore | step | 0.6 | **64.71%** | 53.21% | **58.41%** | **77.52%** |
| 2 | Omnivore | recordings | 0.5 | 40.24% | 69.29% | **50.91%** | 56.83% |
| 3 | SlowFast | step | 0.6 | 36.03% | **87.55%** | 51.05% | 68.30% |
| 4 | SlowFast | recordings | 0.5 | 40.22% | **89.63%** | 55.53% | 58.99% |

---

## Key Observations

### 1. Best Configuration: **Omnivore + Step Split**
- **Test F1**: 58.41% - highest among all configurations
- **Test AUC**: 77.52%
- **Test Precision**: 64.71% (highest)
- With reduced epochs (15), achieves excellent performance with controlled overfitting

### 2. Recordings Split Performance
- **Omnivore+Recordings Test F1**: 50.91%
- **SlowFast+Recordings Test F1**: **55.53%**
- The threshold=0.5 ensures proper recall performance

### 3. Overfitting Analysis

| Config | Epochs | Train Loss | Val Loss | Test Loss | Overfitting? |
|--------|--------|------------|----------|-----------|--------------|
| Omnivore/step | 15 | 0.449 | 1.248 | 1.240 | ⚠️ Moderate (reduced) |
| Omnivore/rec | 15 | 0.425 | 1.413 | 1.587 | ⚠️ Moderate (reduced) |
| SlowFast/step | 30 | 0.938 | 1.013 | 0.967 | ✅ No (stable) |
| SlowFast/rec | 30 | 0.921 | 1.046 | 1.022 | ✅ No (stable) |

**Key Finding**: Reducing Omnivore epochs from 30→15 reduces overfitting while maintaining performance.

### 4. Epoch Recommendations

| Config | Epochs | Recommendation |
|--------|--------|----------------|
| Omnivore/step | 15 | ✅ 15 epochs optimal |
| Omnivore/rec | 15 | ✅ 15 epochs optimal |
| SlowFast/step | 30 | ✅ 30 epochs stable |
| SlowFast/rec | 30 | ✅ 30 epochs stable |

---

## Comparison with MLP/Transformer Baselines

Based on CaptainCook4D paper (Table from colab_quickstart evaluation):

### Step Split (Threshold=0.6)

| Model | Omnivore F1 | SlowFast F1 |
|-------|-------------|-------------|
| MLP (V1) | ~35-40% | ~35-40% |
| Transformer (V2) | ~40-45% | ~40-45% |
| **LSTM (Ours NEW)** | **50.85%** (val) / **58.41%** (test) | **50.93%** (val) / 51.05% (test) |

### Recordings Split (Threshold=0.5)

| Model | Omnivore F1 | SlowFast F1 |
|-------|-------------|-------------|
| MLP (V1) | ~40-45% | ~40-45% |
| Transformer (V2) | ~50-55% | ~50-55% |
| **LSTM (Ours NEW)** | **55.98%** (val) / **50.91%** (test) | **52.56%** (val) / **55.53%** (test) |

---

## Conclusions

### Key Findings:
1. **LSTM on Omnivore/step achieves highest test F1 (58.41%)** - exceeds paper baselines
2. **Correct thresholds are critical**: step=0.6, recordings=0.5
3. **SlowFast configurations are stable** - no overfitting, consistent results
4. **Reduced epochs for Omnivore (15 vs 30)** - maintains performance with less overfitting
5. LSTM successfully captures temporal dependencies in step sequences

### Final Recommendations:
1. **For best results**: Use Omnivore/step (F1=58.41%) with 15 epochs
2. **For recordings split**: Use SlowFast (F1=55.53%) with threshold=0.5
3. **Always use correct thresholds**: step=0.6, recordings=0.5
4. **SlowFast is more stable** but Omnivore achieves higher peak performance

---

## Raw Training Curves Summary

### Configuration 1: Omnivore + Step (15 epochs)
```
Best Val F1: 50.85% (Epoch 15)
Final Test F1: 58.41%
Overfitting: Moderate (train_loss: 0.45, val_loss: 1.25)
```

### Configuration 2: Omnivore + Recordings (15 epochs, th=0.5)
```
Best Val F1: 55.98% (Epoch 15)
Final Test F1: 50.91%
Overfitting: Moderate (train_loss: 0.43, val_loss: 1.41)
```

### Configuration 3: SlowFast + Step (30 epochs)
```
Best Val F1: 50.93% (Epoch 30)
Final Test F1: 51.05%
Overfitting: No (train_loss: 0.94, val_loss: 1.01)
```

### Configuration 4: SlowFast + Recordings (30 epochs, th=0.5)
```
Best Val F1: 52.56% (Epoch 30)
Final Test F1: 55.53%
Overfitting: No (train_loss: 0.92, val_loss: 1.05)
```


