# Mathematical Foundations of CACIS

## Cost-Aware Classification

CACIS (Cost-Aware Classification with Informative Selection) is based on decision-theoretic principles for learning classifiers under example-dependent misclassification costs.

## Overview

Traditional classification systems minimize error rates without considering the real-world costs of different types of errors. CACIS addresses this limitation by:

1. **Example-Dependent Costs**: Allowing different misclassification costs for each example
2. **Real-World Units**: Expressing costs in meaningful units (euros, energy, time)
3. **Full Confusion Matrix**: Considering costs for all cells of the confusion matrix

## Core Components

### 1. Cost-Aware Loss Functions (torch/)

PyTorch implementations of loss functions that incorporate example-wise costs during training.

### 2. Framework-Agnostic Core (core/)

Mathematical implementations that work independently of any specific machine learning framework, providing flexibility and portability.

### 3. Uncertainty Quantification (uncertainty/)

Cost-aware example-wise conformal prediction methods for uncertainty quantification that respect the cost structure of the problem.

## Mathematical Formulation

### Cost Matrix

For each example, we define a cost matrix where entry C[i,j] represents the cost of predicting class j when the true class is i.

### Expected Cost Minimization

The optimal prediction minimizes the expected cost:

```
ŷ = argmin_j Σ_i P(y=i|x) * C[i,j]
```

where P(y=i|x) is the predicted probability of class i given input x.

## References

For more detailed information, see the documentation in the respective module directories.
