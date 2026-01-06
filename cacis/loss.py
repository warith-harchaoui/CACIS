"""
Cost-Aware Classification with Implicit Scores (CACIS).

This module implements CACIS as an **implicit Fenchel–Young loss** for
cost-sensitive classification problems.

The loss is designed for settings where:
- misclassification costs are asymmetric,
- costs may be example-dependent,
- costs have real semantic meaning (€, time, risk, energy, ...).

---------------------------------------------------------------------
Raw CACIS loss
---------------------------------------------------------------------
Given:
- scores f ∈ R^K,
- true label y ∈ {0, …, K−1},
- cost matrix C ∈ R^{K×K},

the raw CACIS loss is defined as:

    ℓ_raw(y, f; C, ε) = Ω*_{C,ε}(f) − f_y

where Ω* is the Fenchel conjugate of a convex regularizer Ω_{C,ε}.

The conjugate is evaluated via an inner optimization problem that is
solved numerically (Frank–Wolfe), but **we do not backpropagate through
that solver**. Gradients w.r.t. scores are defined implicitly.

---------------------------------------------------------------------
Baseline-normalized loss (Option B, τ = 0)
---------------------------------------------------------------------
Raw CACIS values are optimized directly, but they are hard to interpret.
We therefore report a *baseline-normalized* loss for logging only.

Baseline (cost-aware uninformed predictor)
------------------------------------------
If one knows only the cost matrix C (and not x), a natural uninformed
decision rule is to predict the class with minimal expected cost:

    r_k = E_y [ C_{y,k} ]

Baseline scores are defined as:

    f_base = −r

The baseline loss ℓ_base is obtained by evaluating the *same raw CACIS
loss* at f_base.

Normalization (masked mean-of-ratios)
-------------------------------------
For asymmetric problems (e.g. fraud), many samples have zero baseline
risk. Normalizing those is undefined and economically meaningless.

We therefore define:

    ℓ_norm,i = ℓ_raw,i / ℓ_base,i    if ℓ_base,i > 0
             = ignored               otherwise

and report:

    loss_norm = mean_i ℓ_norm,i over {i | ℓ_base,i > 0}

By construction:
- loss_norm → 0 for perfect prediction,
- loss_norm = 1 for the cost-aware uninformed baseline,
- loss_norm < 1 indicates improvement over baseline.

---------------------------------------------------------------------
Design principles
---------------------------------------------------------------------
- The forward pass returns **both** raw and normalized losses.
- Backpropagation is performed **only on the raw loss**.
- The normalized loss is a pure Python float (not differentiable).
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import torch
from torch import Tensor
from torch.nn import Module

# =============================================================================
# Types
# =============================================================================

_EpsilonMode = Literal["constant", "offdiag_mean", "offdiag_median", "offdiag_max"]


# =============================================================================
# Helper functions
# =============================================================================

def _off_diagonal_stat(
    C: Tensor,
    stat: Literal["mean", "median", "max"],
) -> Tensor:
    """
    Compute a statistic of off-diagonal entries of a cost matrix.

    Parameters
    ----------
    C : Tensor
        Cost matrix of shape (K, K) or batch of shape (B, K, K).
    stat : {"mean", "median", "max"}
        Statistic to compute.

    Returns
    -------
    Tensor
        Scalar tensor if C is (K, K),
        tensor of shape (B,) if C is (B, K, K).
    """
    if C.ndim == 2:
        K = C.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        vals = C[mask]
        if stat == "mean":
            return vals.mean()
        if stat == "median":
            return vals.median()
        if stat == "max":
            return vals.max()

    if C.ndim == 3:
        B, K, _ = C.shape
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        mask = mask.unsqueeze(0).expand(B, K, K)
        vals = C[mask].view(B, K * K - K)
        if stat == "mean":
            return vals.mean(dim=1)
        if stat == "median":
            return vals.median(dim=1).values
        if stat == "max":
            return vals.max(dim=1).values

    raise ValueError("C must have shape (K,K) or (B,K,K).")


def _solve_qp_on_simplex(M: Tensor, n_iter: int) -> Tensor:
    """
    Approximate solution of:
        min_{α ∈ Δ_K} αᵀ M α
    using Frank–Wolfe iterations.

    Parameters
    ----------
    M : Tensor
        Tensor of shape (B, K, K).
    n_iter : int
        Number of Frank–Wolfe iterations.

    Returns
    -------
    Tensor
        Approximate minimizer α of shape (B, K).
    """
    B, K, _ = M.shape
    alpha = torch.full((B, K), 1.0 / K, device=M.device, dtype=M.dtype)

    for it in range(n_iter):
        grad = 2.0 * torch.bmm(M, alpha.unsqueeze(-1)).squeeze(-1)
        idx = grad.argmin(dim=1)
        s = torch.zeros_like(alpha)
        s.scatter_(1, idx.unsqueeze(1), 1.0)
        step = 2.0 / (it + 2.0)
        alpha = (1.0 - step) * alpha + step * s

    return alpha


# =============================================================================
# Public API
# =============================================================================

class CACISLossOutput(NamedTuple):
    """Output container for CACISLoss."""
    loss: Tensor
    loss_norm: float
    normalized: bool


class CACISLoss(Module):
    """
    CACIS implicit Fenchel–Young loss with baseline-normalized reporting.
    """

    def __init__(
        self,
        *,
        epsilon_mode: _EpsilonMode = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 2.0,
        epsilon_min: float = 1e-8,
        solver_iter: int = 50,
    ) -> None:
        super().__init__()

        if epsilon_mode == "constant" and epsilon is None:
            raise ValueError("epsilon must be provided when epsilon_mode='constant'.")

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.epsilon_min = epsilon_min
        self.solver_iter = solver_iter

    # ------------------------------------------------------------------

    def _compute_epsilon(self, C: Tensor) -> Tensor:
        if self.epsilon_mode == "constant":
            eps = torch.as_tensor(self.epsilon, device=C.device, dtype=C.dtype)
        else:
            stat = self.epsilon_mode.replace("offdiag_", "")
            eps = _off_diagonal_stat(C, stat)
        return torch.clamp(eps * self.epsilon_scale, min=self.epsilon_min)

    def baseline_scores(self, C: Tensor) -> Tensor:
        if C.ndim == 2:
            return -C.mean(dim=0)
        if C.ndim == 3:
            return -C.mean(dim=1)
        raise ValueError("C must have shape (K,K) or (B,K,K).")

    def _conjugate_term(self, scores: Tensor, C: Tensor, eps: Tensor) -> Tensor:
        B, K = scores.shape
        f_i = (0.5 * scores).unsqueeze(2)
        f_j = (0.5 * scores).unsqueeze(1)
        exponent = -(f_i + f_j + C) / eps.view(-1, 1, 1)
        shift = exponent.amax(dim=(1, 2), keepdim=True)
        logM = exponent - shift
        M = torch.exp(logM)

        with torch.no_grad():
            alpha = _solve_qp_on_simplex(M, self.solver_iter)

        neginf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
        loga = torch.where(alpha > 0, torch.log(alpha), neginf)
        term = loga.unsqueeze(2) + loga.unsqueeze(1) + logM
        logval = torch.logsumexp(term.view(B, K * K), dim=1)

        return -eps * (logval + shift.squeeze())

    def _raw_loss_per_example(self, scores: Tensor, targets: Tensor, C: Tensor) -> Tensor:
        eps = self._compute_epsilon(C)
        conj = self._conjugate_term(scores, C, eps)
        f_y = scores.gather(1, targets.view(-1, 1)).squeeze(1)
        return conj - f_y

    # ------------------------------------------------------------------

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        C: Optional[Tensor] = None,
        normalize: bool = True,
    ) -> CACISLossOutput:

        B, K = scores.shape
        device, dtype = scores.device, scores.dtype

        if C is None:
            C = torch.ones((K, K), device=device, dtype=dtype)
            C.fill_diagonal_(0.0)

        Cb = C.unsqueeze(0).expand(B, -1, -1) if C.ndim == 2 else C

        raw_vec = self._raw_loss_per_example(scores, targets, Cb)
        loss = raw_vec.mean()

        if normalize:
            with torch.no_grad():
                base_scores = self.baseline_scores(Cb)
                base_vec = self._raw_loss_per_example(base_scores, targets, Cb)
                mask = base_vec > 0
                loss_norm = (raw_vec[mask] / base_vec[mask]).mean().item() if mask.any() else 0.0
        else:
            loss_norm = None

        return CACISLossOutput(loss=loss, loss_norm=loss_norm, normalized=normalize)
