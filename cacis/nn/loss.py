"""
CACIS loss: Cost-Aware Classification with Informative Selection.

This module implements CACIS as an implicit Fenchel–Young (FY) loss derived
from an entropy-regularized optimal-transport (OT) energy.

The loss is defined **at the score level** and is model-agnostic: it can be
used with linear models, neural networks, or any model producing class scores.

Key design principles
---------------------
- The loss is differentiable w.r.t. `scores` only.
- The inner optimization (argmin over the simplex) is solved approximately
  and is NOT differentiated through.
- Gradients are defined implicitly via Fenchel–Young theory.
- Cost matrices may be example-dependent and expressed in physical units.

Interpretability note
---------------------
Raw CACIS values are meaningful for optimization, but (just like cross-entropy)
they are most interpretable **relative to a baseline**.

We therefore provide a baseline-normalized variant that is the CACIS analogue
of expressing cross-entropy in base K.

Let:
- \ell_raw be the raw CACIS loss.
- \ell_base be the loss obtained by a *cost-aware uninformed* predictor
  that uses only the cost matrix C (no x).

Then we define the **normalized CACIS**:

    \ell_norm = \ell_raw / (\ell_base + tiny)

This makes the scale directly interpretable:

===========================  ===============================================
Normalized value             Interpretation
===========================  ===============================================
0                            Perfect prediction (\ell_raw -> 0)
< 1                          Better than cost-aware random guessing
1                            Cost-aware random (uninformed) guessing
\ell_uniform / \ell_base     Uniform guessing (all scores equal);
                             equals 1 when C = 1 - I
> 1                          Worse than the cost-aware random baseline
>> 1                         Catastrophically wrong
===========================  ===============================================

What is the cost-aware uninformed baseline?
-------------------------------------------
If you do not know x, but you do know the cost geometry C, a natural summary
of how "dangerous" it is to predict class j is its average risk:

    r_j = mean_i C_{i,j}

We then assign higher preference to lower-risk classes by using a softmin.
Because CACIS already uses \varepsilon as the temperature that converts costs
into exponentials, we use the same \varepsilon for the baseline.

Implementation detail:
We build baseline *scores* as:

    scores_base = -r

and let the CACIS machinery turn those scores into a baseline loss.
This baseline reduces to uniform guessing when C = 1 - I.
"""

from __future__ import annotations

from typing import Optional, Literal

import torch
from torch import Tensor
from torch.nn import Module


# ============================================================
# Utilities
# ============================================================


def off_diagonal_value(
    C: Tensor,
    value: Literal["mean", "median", "max"] = "mean",
) -> Tensor:
    """Compute a statistic of off-diagonal entries of a cost matrix.

    Parameters
    ----------
    C : Tensor, shape (B, K, K) or (K, K)
        Cost matrix or batch of cost matrices.
    value : {"mean", "median", "max"}, default="mean"
        Statistic to compute on off-diagonal entries.

    Returns
    -------
    stat : Tensor, shape (B,) or scalar
        Requested statistic of off-diagonal costs.
    """
    if C.ndim == 2:
        K = C.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        vals = C[mask]
        if value == "mean":
            return vals.mean()
        if value == "median":
            return vals.median()
        if value == "max":
            return vals.max()

    if C.ndim == 3:
        B, K, _ = C.shape
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        mask = mask.unsqueeze(0).expand(B, K, K)
        vals = C[mask].view(B, K * K - K)
        if value == "mean":
            return vals.mean(dim=1)
        if value == "median":
            return vals.median(dim=1).values
        if value == "max":
            return vals.max(dim=1).values

    raise ValueError("C must have shape (K,K) or (B,K,K)")


def solve_qp_simplex(M: Tensor, n_iter: int = 50) -> Tensor:
    """Approximately solve min_{alpha in Δ_K} alpha^T M alpha via Frank–Wolfe.

    IMPORTANT
    ---------
    - This solver is intentionally NOT differentiable.
    - It is used as an implicit layer in a Fenchel–Young loss.
    - Gradients are defined by theory, not by backprop through the solver.

    Parameters
    ----------
    M : Tensor, shape (B, K, K)
        Positive matrix defining the quadratic form.
    n_iter : int, default=50
        Number of Frank–Wolfe iterations.

    Returns
    -------
    alpha : Tensor, shape (B, K)
        Approximate minimizer on the simplex.
    """
    B, K, _ = M.shape
    alpha = torch.full((B, K), 1.0 / K, device=M.device, dtype=M.dtype)

    for it in range(n_iter):
        # Gradient of alpha^T M alpha is 2 M alpha
        grad = 2.0 * torch.bmm(M, alpha.unsqueeze(-1)).squeeze(-1)

        # Linear minimization oracle on simplex: put all mass on argmin
        idx = grad.argmin(dim=1)
        s = torch.zeros_like(alpha)
        s.scatter_(1, idx.unsqueeze(1), 1.0)

        # Standard FW step size
        step = 2.0 / (it + 2.0)
        alpha = (1.0 - step) * alpha + step * s

    return alpha


# ============================================================
# CACIS Loss
# ============================================================


epsilon_modes = ["constant", "offdiag_mean", "offdiag_median", "offdiag_max"]


class CACISLoss(Module):
    """CACIS implicit Fenchel–Young loss with optional baseline normalization.

    The raw CACIS loss is defined as:

        ℓ_raw(y, f; C, ε) = Ω*_{C,ε}(f) - f_y

    where Ω* is (an approximation of) the Fenchel conjugate of an
    entropy-regularized OT energy.

    Parameters
    ----------
    epsilon_mode : {"offdiag_mean", "offdiag_median", "offdiag_max", "constant"},
        Strategy used to compute ε.
    epsilon : float, optional
        Constant ε if epsilon_mode="constant".
    epsilon_scale : float, default=1.0
        Multiplicative scale applied to ε.
    epsilon_min : float, default=1e-8
        Lower bound for numerical stability.
    solver_iter : int, default=50
        Number of Frank–Wolfe iterations for the inner simplex solver.
    """

    def __init__(
        self,
        *,
        epsilon_mode: Literal[epsilon_modes] = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        solver_iter: int = 50,
    ) -> None:
        super().__init__()

        if epsilon_mode == "constant" and epsilon is None:
            raise ValueError("epsilon must be provided when epsilon_mode='constant'")
        if epsilon_mode not in epsilon_modes:
            raise ValueError(f"Invalid epsilon_mode: {epsilon_mode}")

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.epsilon_min = epsilon_min
        self.solver_iter = solver_iter

    # --------------------------------------------------------
    # ε utilities
    # --------------------------------------------------------

    def _compute_epsilon(self, C: Tensor) -> Tensor:
        """Compute ε from the cost matrix."""
        if self.epsilon_mode == "constant":
            eps = torch.as_tensor(self.epsilon, device=C.device, dtype=C.dtype)
        else:
            mode = self.epsilon_mode.replace("offdiag_", "")
            eps = off_diagonal_value(C, mode)

        eps = self.epsilon_scale * eps
        return torch.clamp(eps, min=self.epsilon_min)

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    @staticmethod
    def uniform_scores(scores_like: Tensor) -> Tensor:
        """Return uniform scores (all zeros) with the same shape/device/dtype."""
        return torch.zeros_like(scores_like)

    def baseline_scores(self, C: Tensor) -> Tensor:
        """Return cost-aware *uninformed* baseline scores derived from C.

        We summarize each predicted class j by its average risk:

            r_j = mean_i C_{i,j}

        and return baseline scores f_base = -r.

        Parameters
        ----------
        C : Tensor, shape (B, K, K)
            Batch of cost matrices.

        Returns
        -------
        scores_base : Tensor, shape (B, K)
            Baseline scores.
        """
        if C.ndim != 3:
            raise ValueError("baseline_scores expects C with shape (B, K, K)")
        # Column-wise mean: risk of predicting class j when the true class is unknown.
        r = C.mean(dim=1)  # (B, K)
        return -r

    # --------------------------------------------------------
    # Core loss
    # --------------------------------------------------------

    def _raw_loss(self, scores: Tensor, targets: Tensor, C: Tensor) -> Tensor:
        """Compute the raw CACIS loss per example (shape (B,)).

        Numerical notes
        --------------
        The quadratic form val = alpha^T M alpha can underflow in float32.
        We therefore compute log(val) in log-space via logsumexp using log(M)
        and a masked log(alpha) (alpha may have exact zeros).
        """
        B, K = scores.shape
        eps = self._compute_epsilon(C)  # (B,)

        # Scores enter symmetrically (as in OT dual potentials). To make the
        # resulting FY loss compatible with the standard one-hot linear term
        # (-f_y), we use half-scores on each side.
        #
        # This is also the gauge that restores score-shift invariance:
        # adding a constant to all scores does NOT change the loss.
        f_i = (0.5 * scores).unsqueeze(2)  # (B, K, 1)
        f_j = (0.5 * scores).unsqueeze(1)  # (B, 1, K)

        exponent = -(f_i + f_j + C) / eps.view(-1, 1, 1)  # (B, K, K)

        # Stabilize exp(): subtract the per-example max.
        shift = exponent.amax(dim=(1, 2), keepdim=True)  # (B, 1, 1)
        logM = exponent - shift
        M = torch.exp(logM)

        # Solve inner minimization (implicit layer) without backprop.
        with torch.no_grad():
            alpha = solve_qp_simplex(M, n_iter=self.solver_iter)  # (B, K)

        # log(val) where val = sum_{i,j} alpha_i * M_ij * alpha_j.
        # Compute in log-space to avoid underflow.
        neginf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
        loga = torch.where(alpha > 0, torch.log(alpha), neginf)  # (B, K)
        term = loga.unsqueeze(2) + loga.unsqueeze(1) + logM  # (B, K, K)
        logval = torch.logsumexp(term.view(B, K * K), dim=1)  # (B,)

        # Fenchel conjugate term
        conjugate = -eps * (logval + shift.squeeze())  # (B,)

        # Linear term (one-hot target)
        f_y = scores.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)

        return conjugate - f_y

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def baseline_loss(self, targets: Tensor, C: Tensor) -> Tensor:
        """Compute the cost-aware uninformed baseline loss per example.

        Parameters
        ----------
        targets : Tensor, shape (B,)
            Ground-truth labels.
        C : Tensor, shape (B, K, K)
            Batch of cost matrices.

        Returns
        -------
        loss_base : Tensor, shape (B,)
            Per-example baseline loss.
        """
        scores_base = self.baseline_scores(C)
        return self._raw_loss(scores_base, targets, C)

    def uniform_loss(self, targets: Tensor, C: Tensor) -> Tensor:
        """Compute the loss of uniform guessing (all scores equal) per example."""
        B, K, _ = C.shape
        scores_u = torch.zeros((B, K), device=C.device, dtype=C.dtype)
        return self._raw_loss(scores_u, targets, C)

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        C: Optional[Tensor] = None,
        *,
        normalized: bool = False,
    ) -> Tensor:
        """Compute the CACIS loss.

        Parameters
        ----------
        scores : Tensor, shape (B, K)
            Model scores (logits).
        targets : Tensor, shape (B,)
            Ground-truth labels.
        C : Tensor, shape (B, K, K) or (K, K), optional
            Cost matrices. If None, defaults to 1 - I (cross-entropy regime).
        normalized : bool, default=False
            If True, return baseline-normalized interpretable CACIS.

        Returns
        -------
        loss : Tensor, shape ()
            Scalar loss (mean over batch).
        """
        B, K = scores.shape
        device = scores.device
        dtype = scores.dtype

        # Default to uniform misclassification cost if C is not provided.
        if C is None:
            C = torch.ones((K, K), device=device, dtype=dtype)
            C.fill_diagonal_(0.0)

        # Ensure batch shape.
        if C.ndim == 2:
            C = C.unsqueeze(0).expand(B, -1, -1)
        elif C.ndim != 3:
            raise ValueError("C must have shape (K,K) or (B,K,K)")

        raw = self._raw_loss(scores, targets, C)

        if not normalized:
            return raw.mean()

        base = self.baseline_loss(targets, C)
        tiny = torch.finfo(raw.dtype).tiny
        norm = raw / (base.clamp_min(tiny))
        return norm.mean()
