# CACIS — Cost-Aware Classification with Informative Selection

**CACIS** is an open-source framework for **decision-theoretic classification**
with **example-dependent misclassification costs** expressed in real-world units
(euros, energy, time, risk).

Accuracy is not the objective — **decisions with costs are**.

---

## Motivation

In real-world AI systems:
- misclassification costs vary per example,
- costs are sometimes known only at decision time,
- accuracy is often a misleading proxy for business value.

Standard losses (cross-entropy, weighted CE) do not model this reality.
CACIS provides a principled alternative.

---

## Core Idea

Given data $(x_i,y_i,C_i)$, CACIS learns a calibrated predictive distribution
$q(y \mid x)$
using a **cost-aware Fenchel–Young loss** derived from
entropy-regularized optimal transport.

Decisions are made **only at inference time**, using costs when available.

---

## Inference Policy

CACIS follows a simple and robust decision rule:

- **If a cost matrix $C$ is available at prediction time**:
  use **expected-cost minimization**
  ```math
  \hat{k}(x,C) = \arg\min_k \sum_y q(y \mid x) c_{y,k}(C).
  ```

- **If no cost matrix is available**:
  fall back to **standard probabilistic classification**
  ```math
  \hat{y}(x) = \arg\max_y q(y \mid x).
  ```

This design keeps CACIS usable across heterogeneous deployment contexts.

---

## What CACIS Provides

### Loss
- Convex, proper, score-based loss
- Example-dependent cost matrices during training
- Scale-aware regularization $\varepsilon_i$
- Reduces to cross-entropy for uniform costs

### Training & Prediction
- Model-agnostic (linear models, neural networks)
- Explicit separation between learning and decision

### Uncertainty
- Cost-aware conformal prediction
- Example-wise confidence bounds on realized cost

---

## Project Structure

The project is organized as a standard Python package:

```
cacis/
├── cacis/                  # Main package
│   ├── __init__.py         # Package exports
│   └── nn/                 # Neural Network modules
│       ├── __init__.py     # Exposes CACISLoss
│       ├── loss.py         # CACISLoss class
│       └── ot.py           # Sinkhorn solver
├── examples/               # Usage examples
├── tests/                  # Unit tests
├── README.md               # Project documentation
├── math.md                 # Mathematical foundations
└── requirements.txt        # Dependencies
```

---

## Roadmap

- [x] Mathematical formulation
- [ ] PyTorch loss implementation
- [ ] Training examples
- [ ] sklearn-compatible API
- [ ] Cost-aware conformal uncertainty
- [ ] Tech report and applied dissemination

---

## Positioning

CACIS sits at the intersection of:
- decision theory,
- cost-sensitive learning,
- optimal transport,
- applied machine learning.

It aligns with the **scikit-learn / probabl.ai** philosophy:
clarity, correctness, and practical relevance.

---

## License

BSD 3-Clause License
