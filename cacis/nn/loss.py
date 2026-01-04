"""
CACIS loss: Cost-Aware Classification with Informative Selection.

This module implements the CACIS loss as an implicit Fenchel–Young loss
derived from entropy-regularized optimal transport.

The loss is defined at the score level and is model-agnostic: it can be
used with linear models, neural networks, or any model producing class scores.

Key design principles
---------------------
- The loss is differentiable w.r.t. scores only.
- The inner optimization (argmin over the simplex) is solved approximately
  and is NOT differentiated through.
- Gradients are defined implicitly via Fenchel–Young theory.
- Cost matrices may be example-dependent and expressed in physical units.
"""

from __future__ import annotations

from typing import Optional, Literal

import torch
from torch import Tensor
from torch.nn import Module


def off_diagonal_mean(C: Tensor) -> Tensor:
    """
    Compute the mean of off-diagonal entries of a cost matrix.

    Parameters
    ----------
    C : Tensor, shape (B, K, K) or (K, K)
        Cost matrix or batch of cost matrices.

    Returns
    -------
    eps : Tensor, shape (B,) or scalar
        Mean off-diagonal cost.
    """
    if C.ndim == 2:
        K = C.shape[0]
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        return C[mask].mean()

    elif C.ndim == 3:
        B, K, _ = C.shape
        mask = ~torch.eye(K, dtype=torch.bool, device=C.device)
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return C[mask].view(B, -1).mean(dim=1)

    else:
        raise ValueError("C must have shape (K,K) or (B,K,K)")


def solve_qp_simplex(
    M: Tensor,
    n_iter: int = 50,
) -> Tensor:
    """
    Approximately solve

        min_{alpha in Δ_K}  alpha^T M alpha

    using Frank–Wolfe on the simplex.

    IMPORTANT
    ---------
    - This solver is intentionally *not* differentiable.
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
    device = M.device
    dtype = M.dtype

    # Initialize at uniform distribution
    alpha = torch.full((B, K), 1.0 / K, device=device, dtype=dtype)

    for _ in range(n_iter):
        # Gradient of alpha^T M alpha is 2 M alpha
        grad = 2.0 * torch.bmm(M, alpha.unsqueeze(-1)).squeeze(-1)

        # Linear minimization oracle on simplex: put mass on argmin
        idx = grad.argmin(dim=1)
        s = torch.zeros_like(alpha)
        s.scatter_(1, idx.unsqueeze(1), 1.0)

        # Standard FW step size
        step = 2.0 / (_ + 2.0)
        alpha = (1.0 - step) * alpha + step * s

    return alpha


class CACISLoss(Module):
    """
    CACIS implicit Fenchel–Young loss.

    This loss implements cost-aware classification with example-dependent
    misclassification costs. It is defined as:

        ℓ(y, f; C, ε) = Ω*_{C,ε}(f) - f_y

    where Ω* is the Fenchel conjugate of the Sinkhorn negentropy.

    Notes
    -----
    - The loss is differentiable w.r.t. `scores` only.
    - The inner optimization over the simplex is solved approximately
      and is NOT differentiated through.
    - This is intentional and mathematically correct for FY losses.

    Parameters
    ----------
    epsilon_mode : {"offdiag_mean", "constant"}, default="offdiag_mean"
        Strategy used to compute ε.
    epsilon : float, optional
        Constant ε if epsilon_mode="constant".
    epsilon_scale : float, default=1.0
        Multiplicative scale applied to ε.
    epsilon_min : float, default=1e-8
        Lower bound for numerical stability.
    solver_iter : int, default=50
        Number of iterations for the inner simplex solver.
    """

    def __init__(
        self,
        *,
        epsilon_mode: Literal["offdiag_mean", "constant"] = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        solver_iter: int = 50,
    ) -> None:
        super().__init__()

        if epsilon_mode == "constant" and epsilon is None:
            raise ValueError("epsilon must be provided when epsilon_mode='constant'")

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon
        self.epsilon_scale = epsilon_scale
        self.epsilon_min = epsilon_min
        self.solver_iter = solver_iter

    def _compute_epsilon(self, C: Tensor) -> Tensor:
        """
        Compute ε for each example.

        Parameters
        ----------
        C : Tensor, shape (B, K, K) or (K, K)

        Returns
        -------
        eps : Tensor, shape (B,) or scalar
        """
        if self.epsilon_mode == "constant":
            eps = torch.as_tensor(
                self.epsilon,
                device=C.device,
                dtype=C.dtype,
            )
        else:
            eps = off_diagonal_mean(C)

        eps = self.epsilon_scale * eps
        return torch.clamp(eps, min=self.epsilon_min)

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        C: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute the CACIS loss.

        Parameters
        ----------
        scores : Tensor, shape (B, K)
            Model scores (logits).
        targets : Tensor, shape (B,)
            Ground-truth labels.
        C : Tensor, shape (B, K, K) or (K, K), optional
            Cost matrices. If None, defaults to 1 - I (cross-entropy regime).

        Returns
        -------
        loss : Tensor, shape ()
            Scalar loss.
        """
        B, K = scores.shape
        device = scores.device
        dtype = scores.dtype

        # Default to uniform misclassification cost if C is not provided
        if C is None:
            C = torch.ones((K, K), device=device, dtype=dtype)
            C.fill_diagonal_(0.0)

        # Ensure batch shape
        if C.ndim == 2:
            C = C.unsqueeze(0).expand(B, -1, -1)

        eps = self._compute_epsilon(C)  # shape (B,)

        # Build kernel matrix M_ij = exp(-(f_i + f_j + c_ij) / eps)
        f_i = scores.unsqueeze(2)
        f_j = scores.unsqueeze(1)

        exponent = -(f_i + f_j + C) / eps.view(-1, 1, 1)

        # Numerical stabilization
        shift = exponent.amax(dim=(1, 2), keepdim=True)
        M = torch.exp(exponent - shift)

        # Solve inner minimization (implicit layer)
        with torch.no_grad():
            alpha = solve_qp_simplex(M, n_iter=self.solver_iter)

        # Compute quadratic form alpha^T M alpha
        val = torch.bmm(
            alpha.unsqueeze(1),
            torch.bmm(M, alpha.unsqueeze(2)),
        ).squeeze()

        # Fenchel conjugate
        conjugate = -eps * (torch.log(val + 1e-12) + shift.squeeze())

        # Subtract correct class score
        f_y = scores.gather(1, targets.view(-1, 1)).squeeze()

        loss = conjugate - f_y
        return loss.mean()
