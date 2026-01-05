# CACIS on Fraud Detection (IEEE-CIS)

## Overview

This document presents a complete, end-to-end implementation of **Cost-Aware Classification
with Implicit Scores (CACIS)** applied to a realistic **fraud detection** setting inspired
by the IEEE-CIS Fraud Detection dataset.

The goal is not to optimize accuracy, but to **minimize expected economic cost** under
asymmetric, example-dependent misclassification costs.

This makes fraud detection a *canonical* use case for CACIS.

---

## Problem Setting

Each transaction is represented by a triplet:

(x_i, y_i, C_i)

- x_i: transaction features
- y_i ∈ {0,1}: fraud label (1 = fraud)
- C_i ∈ R^{2×2}: example-dependent cost matrix

The classifier does **not** observe costs during training as features; costs are used
only in the **loss and decision rule**.

---

## Cost Modeling

### False Negative (Missed Fraud)

If a fraudulent transaction is accepted:

c_i^FN = α × TransactionAmt_i

Default: α = 1

This corresponds to direct financial loss.

---

### False Positive (Manual Review)

If a legitimate transaction is flagged:

c^FP = t × w

Where:
- t: average review time (minutes)
- w: analyst hourly wage

Default:
- t = 5 minutes
- w = 60 €/hour

c^FP = 5 €

This models **human labor cost**, not model error.

---

### Per-example Cost Matrix

For each transaction i:

C_i = [[0, c^FP],
       [c_i^FN, 0]]

This matrix is:
- asymmetric
- example-dependent
- known at prediction time

---

## Normalization

Raw costs vary by orders of magnitude.
For numerical stability and interpretability, we normalize:

C_i_tilde = C_i / median(c^FN)

This yields a **normalized CACIS loss** with interpretation:

- 0 → perfect decisions
- 1 → cost-aware random guessing
- < 1 → better than random

---

## Model

We deliberately use a **simple model**:
- Logistic regression (PyTorch)

Rationale:
- Interpretability
- Strong baselines
- CACIS already adds sophistication

---

## Training Objective

Given model scores f(x_i), CACIS minimizes expected economic cost.

No threshold tuning is required.

---

## Inference Rule

At prediction time, costs are known.
The decision is:

argmin_k E[c_{Y,k} | x_i, C_i]

This adapts per transaction, unlike global thresholds.

---

## Evaluation Metrics

Primary metric:
- **Total incurred cost (€)**

Secondary:
- Normalized CACIS loss
- ROC-AUC (sanity check only)

---

## Download

```bash
rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true
mkdir ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip
unzip ieee-fraud-detection.zip -d ieee-fraud-detection
```

---

## Conclusion

Fraud detection is not a benchmark problem — it is a **decision problem**.
CACIS provides the correct abstraction.
