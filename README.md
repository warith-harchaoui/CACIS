# CACIS — Cost-Aware Classification with Informative Selection

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**CACIS** is an open-source framework for **decision-theoretic classification** with **example-dependent misclassification costs** expressed in real-world units (euros, energy, time, risk).

> **Accuracy is not the objective — decisions with costs are.**

---

## 💡 Motivation

In real-world AI systems (healthcare, finance, robotics):
- **Misclassification costs vary per example**: A false negative for a rare disease is costlier than for a common cold.
- **Costs are dynamic**: Costs might only be known at decision time (e.g., market prices).
- **Accuracy is a proxy**: Standard metrics often fail to capture the true business or safety impact of a decision.

Standard losses like Cross-Entropy assume uniform costs. CACIS provides a principled alternative derived from **Optimal Transport** and **Fenchel–Young Losses**.

---

## 🧠 Core Idea

Given data $(x_i, y_i, C_i)$, CACIS learns a calibrated predictive distribution $q(y \mid x)$ using a **cost-aware Fenchel–Young loss** regularized by entropy-regularized optimal transport.

### Inference Policy

CACIS decouples learning from decision-making, allowing flexibility at inference time:

1. **If a cost matrix $C$ is available**: use **expected-cost minimization**
   ```math
   \hat{k}(x, C) = \arg\min_k \sum_y q(y \mid x) \; c_{y,k}(C).
   ```
2. **If no cost matrix is available**: fall back to **standard probablistic classification**
   ```math
   \hat{y}(x) = \arg\max_y q(y \mid x).
   ```

For a deep dive into the math, see [math.md](math.md).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/warith-harchaoui/cacis.git
cd cacis

# Install the package
pip install -e .
```

### Basic Usage

```python
import torch
from cacis import CACISLoss

# Model scores (B, K)
logits = torch.randn(8, 10, requires_grad=True) 
# Ground truth labels (B,)
labels = torch.randint(0, 10, (8,))
# Example-dependent cost matrices (B, K, K)
costs = torch.rand(8, 10, 10)

criterion = CACISLoss()
loss = criterion(logits, labels, C=costs)
loss.backward()
```

---

## 🖼️ Featured Demo: ResNet on CIFAR-10

We provide a comprehensive example using **fastText** semantic embeddings to define costs on CIFAR-10. This demo shows how mistakes between "similar" classes (e.g., Cat vs Dog) are penalized less than mistakes between "distant" classes (e.g., Cat vs Truck).

```bash
python image_classification.py
```

**What this demo provides:**
- Automatic download of fastText vectors.
- Semantic cost matrix generation based on word embeddings.
- Training a ResNet18 with CACIS loss.
- **Real-time visualizations**:
  - `images/cost_matrix.png`: The semantic distance structure.
  - `images/loss_trajectory.png`: Optimization progress (normalized CACIS loss).
  - `images/confusion_matrix_epoch_N.png`: Class-wise performance.
  - `images/confusion_matrix_grouped_epoch_N.png`: High-level "Animal vs Vehicle" performance.

---

## 📂 Project Structure

```text
cacis/
├── cacis/                  # Core package
│   ├── nn/                 # Neural Network modules
│   │   ├── __init__.py
│   │   └── loss.py         # CACISLoss implementation
│   └── __init__.py         # Public API
├── image_classification.py # Image classification demo
├── fraud_detection.py      # Fraud detection demo
├── tests/                  # Unit tests
├── utils.py                # Shared utilities
├── math.md                 # Mathematical derivations
├── setup.py                # Package configuration
└── requirements.txt        # Dependencies
```

---

## 🗺️ Roadmap

- [x] Mathematical formulation & Fenchel–Young derivation
- [x] PyTorch `CACISLoss` implementation
- [x] Comprehensive training examples (CIFAR-10 / fastText)
- [ ] scikit-learn compatible `CACISClassifier`
- [ ] Cost-aware conformal uncertainty
- [ ] Technical report / Whitepaper

---

## ⚖️ License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
