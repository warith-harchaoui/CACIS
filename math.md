# CACIS — Cost-Aware Classification with Informative Selection
## Mathematical Foundations

This document provides a self-contained mathematical description of the **CACIS** framework.
It is written for applied ML researchers, engineers, and decision scientists.

---

## 1. Problem Setting

We observe i.i.d. triples
```math
(X, C, Y) \sim \mathcal{D},
```
where:
- $Y \in \{1, \dots, K\}$ is the true class,
- $C \in \mathbb{R}_+^{K \times K}$ is a misclassification cost matrix with $c_{y,y}=0$,
- $C$ may vary across examples and **may or may not be available at prediction time**.

Costs are expressed in physical or business units (euros, energy, time, risk).

The objective is decision-theoretic:
```math
R(k \mid x, C)
=
\mathbb{E}[c_{Y,k}(C) \mid x, C].
```

---

## 2. Bayes-Optimal Decision Rule

If the true conditional distribution is $p(y \mid x,C)$, the Bayes-optimal action is
```math
k^*(x,C)
\in
\arg\min_k
\sum_{y=1}^K
p(y \mid x,C)\,c_{y,k}(C).
```

This rule is optimal **only when the cost matrix $C$ is known at decision time**.

---

## 3. Score-Based Prediction

A model produces scores
```math
f(x) \in \mathbb{R}^K.
```

From scores, we derive a predictive distribution
$q(y \mid x) \in \Delta_K$
using a proper scoring rule.
Learning and decision are strictly separated.

---

## 4. Fenchel–Young Losses

Let $\Omega : \Delta_K \to \mathbb{R} \cup \{+\infty\}$ be convex.
Its conjugate is
```math
\Omega^*(f)
=
\sup_{\alpha \in \Delta_K}
\langle \alpha, f \rangle - \Omega(\alpha).
```

The Fenchel–Young loss is
```math
\ell_\Omega(y,f) = \Omega^*(f) - f_y,
```
with gradient
```math
\nabla_f \ell_\Omega = q - \delta_y.
```

---

## 5. Cost-Aware Sinkhorn Negentropy

Define
```math
\Omega_{C,\varepsilon}(\alpha)
=
-\tfrac12\,\mathrm{OT}_{C,\varepsilon}(\alpha,\alpha),
```
where
```math
\mathrm{OT}_{C,\varepsilon}(\alpha,\beta)
=
\min_{\pi \in U(\alpha,\beta)}
\langle \pi, C \rangle
+
\varepsilon\,\mathrm{KL}(\pi \| \alpha \otimes \beta).
```

The regularization $\varepsilon$ has the same unit as the costs.
Only the ratio $C/\varepsilon$ matters.

---

## 6. Conjugate and CACIS Loss

The conjugate is
```math
\Omega_{C,\varepsilon}^*(f)
=
-\varepsilon
\log
\min_{\alpha \in \Delta_K}
\sum_{i,j}
\alpha_i\alpha_j
\exp\Big(
-\tfrac{f_i+f_j+c_{ij}}{\varepsilon}
\Big).
```

The CACIS loss is
```math
\ell(y,f;C,\varepsilon)
=
\Omega_{C,\varepsilon}^*(f) - f_y.
```

This loss is convex in the scores and differentiable.

---

## 7. Invariances

- **Row-shift invariance**:
  $c_{y,k} \mapsto c_{y,k}+b_y$ leaves the loss unchanged.
- **Scale invariance**:
  $(C,\varepsilon) \mapsto (aC,a\varepsilon)$ leaves the loss unchanged.

---

## 8. Limiting Cases

- $\varepsilon \to \infty$: softmax cross-entropy.
- $\varepsilon \to 0$: hard cost minimization.

---

## 9. Inference and Decision Policy

CACIS supports two inference regimes.

### 9.1 Cost-aware decision (C available)

When a cost matrix $C$ is available at prediction time, CACIS uses
expected-cost minimization:
```math
\hat{k}(x,C)
=
\arg\min_k
\sum_y q(y \mid x)\,c_{y,k}(C).
```

This recovers the Bayes-optimal decision rule.

### 9.2 Fallback decision (C unavailable)

When $C$ is **not** available at prediction time, CACIS falls back to
standard probabilistic classification:
```math
\hat{y}(x) = \arg\max_y q(y \mid x).
```

In this regime, CACIS should be interpreted as learning a
**cost-shaped predictive distribution** rather than guaranteeing
per-example cost optimality.

This dual policy allows CACIS to remain usable in realistic deployment settings
where costs may be partially observed or context-dependent.
