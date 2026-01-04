# CACIS — Cost-Aware Classification with Informative Selection
## Mathematical Foundations

This document provides a self-contained mathematical description of the **CACIS** framework.
It is written for applied ML researchers, engineers, and decision scientists.

---

## 1. Problem Setting

We operate in a supervised learning setting where decisions have asymmetric, instance-dependent costs.
We observe independent and identically distributed (i.i.d.) triples:

```math
(X, C, Y) \sim \mathcal{D},
```

where:
- $X \in \mathcal{X}$ denotes the input features (e.g., images, vectors).
- $Y \in \{1, \dots, K\}$ is the true class label.
- $C \in \mathbb{R}_+^{K \times K}$ is the **misclassification cost matrix**.
  - The entry $c_{y,k}$ represents the cost of predicting class $k$ when the true class is $y$.
  - We assume zero cost for correct classification: $c_{y,y}=0$ for all $y$.
  - Crucially, $C$ may vary across examples and **may or may not be available at prediction time**.

Costs are expressed in physical or business units (e.g., euros, joules, seconds, risk abstract units).

### Objective

The objective is decision-theoretic. We seek a decision rule $k(\cdot)$ that minimizes the expected conditional risk:

```math
R(k \mid x, C)
=
\mathbb{E}[c_{Y,k}(C) \mid x, C]
=
\sum_{y=1}^K p(y \mid x, C) \, c_{y,k}(C).
```

---

## 2. Bayes-Optimal Decision Rule

If the true conditional distribution $p(y \mid x, C)$ were known, the optimal action (Bayes rule) minimizes the posterior expected cost:

```math
k^*(x,C) = \arg\min_{k \in \{1, \dots, K\}}
\sum_{y=1}^K
p(y \mid x,C)\,c_{y,k}(C).
```

**Key Constraint:** This rule is optimal **only when the cost matrix $C$ is known at decision time**. In many realistic scenarios, $C$ is observed during training/calibration but might be latent or only partially observed during inference.

---

## 3. Score-Based Prediction

We employ a **score-based model** (e.g., a neural network) that maps inputs to a vector of scores:

```math
f(x) \in \mathbb{R}^K.
```

From these scores, we derive a predictive probability distribution $q(y \mid x) \in \Delta_K$ over the $K$ classes, where $\Delta_K = \{p \in \mathbb{R}^K \mid p \ge 0, \sum p_i = 1\}$ is the probability simplex.

The mapping from scores $f$ to probabilities $q$ is determined by a **proper scoring rule** or, equivalently, by the choice of a regularization function $\Omega$.

---

## 4. Fenchel–Young Losses

CACIS leverages the framework of **Fenchel–Young metrics**.
Let $\Omega : \Delta_K \to \mathbb{R} \cup \{+\infty\}$ be a strictly convex regularization function (e.g., negative entropy).

### Fenchel Conjugate
The Fenchel conjugate of $\Omega$, denoted $\Omega^*$, is defined on the score space $\mathbb{R}^K$ as:

```math
\Omega^*(f)
=
\sup_{\alpha \in \Delta_K}
\big( \langle \alpha, f \rangle - \Omega(\alpha) \big).
```

From convex analysis, the gradient of the conjugate generally yields the optimal primal vector (the predicted probability distribution):
```math
\nabla_f \Omega^*(f) = \hat{y}(f) \in \Delta_K.
```

### The Loss Function
The associated Fenchel–Young loss between a true label $y$ (represented as a one-hot vector $\delta_y$) and the score vector $f$ is:

```math
\ell_\Omega(y,f) = \Omega^*(f) - f_y + \Omega(\delta_y).
```
*Note: The term $\Omega(\delta_y)$ is usually constant (often 0) for standard regularizers and is omitted in optimization.*

Thus, considering the effective part:
```math
\ell_\Omega(y,f) = \Omega^*(f) - f_y.
```

The gradient of this loss with respect to the scores drives the learning:
```math
\nabla_f \ell_\Omega(y, f) = \nabla \Omega^*(f) - \delta_y = q(y \mid x) - \delta_y.
```
This ensures that at optimality, the predicted distribution $q$ matches the empirical target distribution.

---

## 5. Cost-Aware Sinkhorn Negentropy

Standard classification typically uses the **Shannon entropy** $\Omega(\alpha) = \sum \alpha_i \log \alpha_i$, leading to the Softmax function and Cross-Entropy loss. However, Shannon entropy is isotropic—it does not account for the relational structure of the classes defined by $C$.

CACIS introduces the **Cost-Aware Sinkhorn Negentropy**:

```math
\Omega_{C,\varepsilon}(\alpha)
=
-\frac{1}{2}\,\mathrm{OT}_{C,\varepsilon}(\alpha,\alpha),
```

### Self-Transport Objective
Here, $\mathrm{OT}_{C,\varepsilon}$ represents the entropic Optimal Transport cost between a distribution and itself:

```math
\mathrm{OT}_{C,\varepsilon}(\alpha,\beta)
=
\min_{\pi \in U(\alpha,\beta)}
\left( 
\langle \pi, C \rangle
+
\varepsilon\,\mathrm{KL}(\pi \| \alpha \otimes \beta)
\right),
```
where:
- $U(\alpha, \beta) = \{ \pi \in \mathbb{R}_+^{K \times K} \mid \pi \mathbf{1} = \alpha, \pi^\top \mathbf{1} = \beta \}$ is the polytope of joint couplings.
- $\mathrm{KL}$ is the Kullback–Leibler divergence.
- $\alpha \otimes \beta$ is the product measure (independent coupling).
- $\varepsilon > 0$ is a regularization parameter (temperature) with the same physical unit as costs $C$.

**Intuition**: By defining the regularizer via self-transport with cost $C$, $\Omega_{C,\varepsilon}$ penalizes distributions that place mass on pairs $(i,j)$ with high cost $c_{ij}$. It induces a geometry on the probability simplex that reflects the cost structure.

---

## 6. Conjugate and CACIS Loss

Deriving the Fenchel conjugate of the Sinkhorn negentropy leads to the **CACIS Loss**.

### Closed-Form Conjugate
For the specific choice of $\Omega_{C,\varepsilon}$, the conjugate function admits a variational form (related to the log-partition function of the cost-aware exponential family):

```math
\Omega_{C,\varepsilon}^*(f)
=
-\varepsilon
\log \left(
\min_{\alpha \in \Delta_K}
\sum_{i=1}^K \sum_{j=1}^K
\alpha_i \alpha_j
\exp\left(
-\frac{f_i + f_j + c_{ij}}{\varepsilon}
\right)
\right).
```

### The Loss Expression
Substituting this into the general FY definition yields the CACIS loss:

```math
\ell(y,f;C,\varepsilon)
=
\Omega_{C,\varepsilon}^*(f) - f_y.
```

### Properties
1.  **Convexity**: The loss is convex in $f$.
2.  **Differentiability**: It is smooth due to the $\varepsilon$-smoothing (entropic term).
3.  **Consistency**: Minimizing this loss recovers the true conditional probabilities if the model capacity is sufficient.

---

## 7. Invariances

The CACIS loss exhibits important invariance properties that are crucial for numerical stability and consistent behavior across different cost scales.

- **Row-shift invariance (Score Translation)**:
  Adding a constant to the costs for a given true class $y$ does not change the optimal decision.
  Transformation: $c_{y,k} \mapsto c_{y,k} + b_y$.
  Result: The loss remains unchanged (up to constants irrelevant to gradients w.r.t $f$).

- **Scale invariance**:
  Scaling both the costs and the temperature $\varepsilon$ by the same factor preserves the geometry.
  Transformation: $(C, \varepsilon) \mapsto (\lambda C, \lambda \varepsilon)$ for $\lambda > 0$.
  Result: The optimization landscape is merely scaled; optimality conditions are preserved.

---

## 8. Limiting Cases

The parameter $\varepsilon$ interpolates between hard cost handling and standard entropy regularization.

- **$\varepsilon \to \infty$ (Isotropic Limit)**:
  The cost structure washes out. The transport plan becomes maximizing entropy (independent coupling). The regularizer approaches Shannon entropy, and the loss approaches **Softmax Cross-Entropy**.

- **$\varepsilon \to 0$ (Hard Limit)**:
  The entropy term vanishes. We approach "hard" optimal transport. The loss penalizes based on the exact cost geometry, approaching a **Structured Hinge Loss** or hard cost minimization objective.

---

## 9. Inference and Decision Policy

CACIS supports a flexible inference strategy depending on information availability.

### 9.1 Cost-aware decision (C available)

When the precise cost matrix $C$ is available at prediction time (e.g., dynamic market prices, patient-specific risks):

```math
\hat{k}(x,C)
=
\arg\min_k \mathbb{E}_{y \sim q(\cdot|x)} [c_{y,k}(C)]
=
\arg\min_k \sum_{y=1}^K q(y \mid x)\,c_{y,k}(C).
```
This is the **Bayes-optimal decision rule** plug-in estimator. It takes the predicted probabilities $q$ and minimizes expected risk using the *current* costs.

### 9.2 Fallback decision (C unavailable)

When $C$ is **not** available at prediction time (e.g., typical classification deployment):

```math
\hat{y}(x) = \arg\max_y q(y \mid x).
```
Here, the model predicts the most likely class.

**Interpretation**: In this regime, because the training objective $\Omega_{C,\varepsilon}$ was shaped by the cost distribution, the probability $q(y \mid x)$ is a **"cost-shaped"** distribution. It is not necessarily the true frequency of classes, but a distribution twisted such that its mode aligns with cost-sensitive decisions on average.
