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

## 🧠 Scientific Foundations

CACIS is built upon the theory of [Geometric Losses for Distributional Learning](https://arxiv.org/abs/1905.06005). It leverages entropic Optimal Transport (Sinkhorn) to shape the probability simplex according to the cost geometry.


By regularizing the learning process with a cost-aware Sinkhorn negentropy, CACIS ensures that the model learns a distribution that is naturally "twisted" toward cost-effective decisions.

You can read the math behind CACIS in the [math.md](math.md) file.

---

## 🚀 Quick Start

### Installation

Please read [conda install instructions](https://harchaoui.org/warith/4ml) first

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
# Returns (raw_loss, normalized_loss, is_normalized)
output = criterion(logits, labels, C=costs)
loss = output.loss
loss.backward()
```

---

## 🪩 Featured Demos

### 1. ResNet on CIFAR-10 (Semantic Costs)

We use **fastText** semantic embeddings to define costs on CIFAR-10. This demo shows how mistakes between "similar" classes (e.g., Cat vs Dog) are penalized less than mistakes between "distant" classes (e.g., Cat vs Truck) thanks to fastText semantic similarities.

```bash
# Standard run
python image_classification.py

# Run with slow normalized CACIS reporting for better interpretation
python image_classification.py --normalization
```

![Loss trajectory](assets/image_loss_trajectory.png)

### 2. IEEE-CIS Fraud Detection (Economic Costs)

A tabular data demo on the Kaggle IEEE-CIS Fraud Detection dataset, where costs are directly proportional to transaction amounts.

Download Kaggle IEEE-CIS Fraud Detection dataset:
```bash
mkdir ieee-fraud-detection
wget -c http://deraison.ai/ai/ieee-fraud-detection.zip
unzip ieee-fraud-detection.zip -d ieee-fraud-detection
```

Usage:
```bash
python fraud_detection.py
```

![Loss trajectory](assets/fraud_loss_trajectory.png)

---

## 📂 Project Structure

```text
cacis/
├── cacis/                  # Core package
│   ├── nn/                 # Neural Network submodules
│   │   └── __init__.py
│   ├── loss.py             # CACISLoss implementation
│   ├── utils.py            # Shared utilities (logging, devices, plotting)
│   └── __init__.py         # Public API (CACISLoss, utils)
├── image_classification.py # Image classification demo
├── fraud_detection.py      # Fraud detection demo
├── tests/                  # Unit tests
├── math.md                 # Mathematical derivations (Deep dive)
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
- [ ] pip installable package on PyPI
- [ ] Technical report / Whitepaper

---

## 📚 References

If you use CACIS in your research, please cite:

> Arthur Mensch, Mathieu Blondel, Gabriel Peyré. **Geometric Losses for Distributional Learning**. *arXiv preprint arXiv:1905.06005*, 2019. [[Paper]](https://arxiv.org/abs/1905.06005)

```bibtex
@article{mensch2019geometric,
  title={Geometric Losses for Distributional Learning},
  author={Mensch, Arthur and Blondel, Mathieu and Peyr{\'e}, Gabriel},
  journal={arXiv preprint arXiv:1905.06005},
  year={2019}
}
```

---

## ⚖️ License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
