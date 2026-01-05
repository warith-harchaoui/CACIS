"""
Cost-Aware Classification with Implicit Scores (CACIS).

This module implements CACIS as an **implicit Fenchel–Young loss** derived from
an entropy-regularized optimal-transport (OT) energy.

The key idea is: given model scores ``scores`` and a misclassification cost
matrix ``C``, CACIS defines a convex regularizer Ω_{C,ε} on the simplex and uses
its Fenchel conjugate Ω*_{C,ε} to produce a **differentiable** (w.r.t. scores)
loss:

    ℓ_raw(y, f; C, ε) = Ω*_{C,ε}(f) - f_y

where ``f_y`` is the score of the true class.

Why "implicit"?
---------------
The conjugate term Ω* is evaluated by solving an *inner* minimization over the
probability simplex. We solve that inner problem approximately (Frank–Wolfe)
and **do not backprop through the solver**. The resulting gradient w.r.t.
scores is defined implicitly by Fenchel–Young theory.

Interpretability: baseline-normalized CACIS
-------------------------------------------
Raw CACIS values are meaningful for optimization, but are best interpreted
relative to a baseline (like cross-entropy is often interpreted relative to a
uniform baseline).

We therefore provide a **baseline-normalized** quantity:

    ℓ_norm = ℓ_raw / (ℓ_base + tiny)

where ℓ_base is the loss obtained by a *cost-aware uninformed* predictor that
uses only the cost matrix C (no x).

By construction:

- ℓ_norm → 0 for perfect prediction (ℓ_raw → 0)
- ℓ_norm = 1 for cost-aware guessing using only C

Important design choice (as requested)
--------------------------------------
This implementation returns **both** the raw loss and the normalized loss in
a *single* forward call, and **does not** expose a ``normalized=...`` boolean.

The returned objects are:
- ``loss``      : a scalar Torch tensor (for backprop), equal to mean(ℓ_raw)
- ``loss_norm`` : a Python float (for logging), equal to mean(ℓ_raw/ℓ_base)

The normalized quantity is *never* part of the autograd graph.

Cost matrices
-------------
- C may be global (shape (K, K)) or example-dependent (shape (B, K, K)).
- Costs may be expressed in physical/business units (€, energy, time, risk, ...).
- Diagonal is assumed to be 0 (no cost for correct prediction), but we do not
  enforce this beyond common-sense defaults.

"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import torch
from torch import Tensor
from torch.nn import Module


# =============================================================================
# Helper functions
# =============================================================================

_EpsilonMode = Literal["constant", "offdiag_mean", "offdiag_median", "offdiag_max"]


def _off_diagonal_stat(
    C: Tensor,
    stat: Literal["mean", "median", "max"] = "mean",
) -> Tensor:
    """
    Compute a statistic of the off-diagonal entries of a cost matrix.

    Parameters
    ----------
    C:
        Cost matrix of shape ``(K, K)`` or a batch ``(B, K, K)``.
    stat:
        Statistic over off-diagonal entries.

    Returns
    -------
    Tensor
        - scalar tensor if ``C`` is (K, K)
        - shape (B,) tensor if ``C`` is (B, K, K)

    Notes
    -----
    This is used to define the temperature/regularization parameter ε as a
    typical scale of off-diagonal costs.
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
        raise ValueError(f"Invalid stat: {stat}")

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
        raise ValueError(f"Invalid stat: {stat}")

    raise ValueError("C must have shape (K, K) or (B, K, K).")


def _solve_qp_on_simplex(M: Tensor, n_iter: int = 50) -> Tensor:
    """
    Approximately solve::

        min_{α ∈ Δ_K}  αᵀ M α

    via Frank–Wolfe on the simplex.

    Parameters
    ----------
    M:
        Tensor of shape ``(B, K, K)`` (should be nonnegative).
    n_iter:
        Number of Frank–Wolfe iterations.

    Returns
    -------
    Tensor
        Approximate minimizer α of shape ``(B, K)``.

    Notes
    -----
    This solver is intentionally used under ``torch.no_grad()`` inside the loss.
    We do *not* backpropagate through this optimization routine.
    """
    if M.ndim != 3:
        raise ValueError("M must have shape (B, K, K).")

    B, K, _ = M.shape
    alpha = torch.full((B, K), 1.0 / K, device=M.device, dtype=M.dtype)

    for it in range(n_iter):
        # Gradient of αᵀ M α is 2 M α
        grad = 2.0 * torch.bmm(M, alpha.unsqueeze(-1)).squeeze(-1)

        # Linear minimization oracle on the simplex: put mass on argmin coordinate
        idx = grad.argmin(dim=1)
        s = torch.zeros_like(alpha)
        s.scatter_(1, idx.unsqueeze(1), 1.0)

        # Standard Frank–Wolfe step-size
        step = 2.0 / (it + 2.0)
        alpha = (1.0 - step) * alpha + step * s

    return alpha


# =============================================================================
# Public API
# =============================================================================

class CACISLossOutput(NamedTuple):
    """
    Output of :class:`CACISLoss`.

    Attributes
    ----------
    loss:
        Scalar torch tensor (mean raw loss). Backpropagate through this only.
    loss_norm:
        Python float (mean baseline-normalized loss). For logging/plotting only.
    """
    loss: Tensor
    loss_norm: float


class CACISLoss(Module):
    """
    CACIS implicit Fenchel–Young loss with baseline-normalized reporting.

    Parameters
    ----------
    epsilon_mode:
        Strategy used to compute ε:
        - ``"offdiag_mean"`` (default): ε := mean off-diagonal cost(s)
        - ``"offdiag_median"``: ε := median off-diagonal cost(s)
        - ``"offdiag_max"``: ε := max off-diagonal cost(s)
        - ``"constant"``: ε := provided ``epsilon``
    epsilon:
        Constant ε if ``epsilon_mode="constant"``.
    epsilon_scale:
        Multiplicative scale applied to ε.
    epsilon_min:
        Lower bound for numerical stability.
    solver_iter:
        Frank–Wolfe iterations for the inner simplex solver.

    Notes
    -----
    *Backpropagates only through the raw loss.*

    The normalized loss is computed as a pure float, detached from autograd,
    and is intended solely for human interpretation.
    """

    def __init__(
        self,
        *,
        epsilon_mode: _EpsilonMode = "offdiag_mean",
        epsilon: Optional[float] = None,
        epsilon_scale: float = 1.0,
        epsilon_min: float = 1e-8,
        solver_iter: int = 50,
    ) -> None:
        super().__init__()

        if epsilon_mode == "constant" and epsilon is None:
            raise ValueError("epsilon must be provided when epsilon_mode='constant'.")
        if epsilon_mode not in ("constant", "offdiag_mean", "offdiag_median", "offdiag_max"):
            raise ValueError(f"Invalid epsilon_mode: {epsilon_mode}")

        if epsilon_scale <= 0:
            raise ValueError("epsilon_scale must be > 0.")
        if epsilon_min <= 0:
            raise ValueError("epsilon_min must be > 0.")
        if solver_iter <= 0:
            raise ValueError("solver_iter must be > 0.")

        self.epsilon_mode = epsilon_mode
        self.epsilon = epsilon
        self.epsilon_scale = float(epsilon_scale)
        self.epsilon_min = float(epsilon_min)
        self.solver_iter = int(solver_iter)

        # Cache for constant (K,K) cost matrices to avoid recomputing baseline
        # conjugate term at every step.
        self._cache_C_id: Optional[int] = None
        self._cache_device: Optional[torch.device] = None
        self._cache_dtype: Optional[torch.dtype] = None
        self._cache_eps: Optional[float] = None
        self._cache_solver_iter: Optional[int] = None
        self._cache_scores_base: Optional[Tensor] = None   # (K,)
        self._cache_conj_base: Optional[Tensor] = None     # scalar tensor

    # -------------------------------------------------------------------------
    # ε (temperature) utilities
    # -------------------------------------------------------------------------

    def _compute_epsilon(self, C: Tensor) -> Tensor:
        """
        Compute ε from the cost matrix C.

        Returns a tensor of shape:
        - scalar if C is (K,K)
        - (B,) if C is (B,K,K)
        """
        if self.epsilon_mode == "constant":
            eps = torch.as_tensor(self.epsilon, device=C.device, dtype=C.dtype)
        else:
            stat = self.epsilon_mode.replace("offdiag_", "")
            eps = _off_diagonal_stat(C, stat=stat)  # scalar or (B,)

        eps = eps * self.epsilon_scale
        return torch.clamp(eps, min=self.epsilon_min)

    # -------------------------------------------------------------------------
    # Baseline (cost-aware uninformed) predictor
    # -------------------------------------------------------------------------

    def baseline_scores(self, C: Tensor) -> Tensor:
        """
        Compute cost-aware *uninformed* baseline scores from the cost matrix.

        If you do not know x but do know the cost geometry C, a natural summary
        of how "dangerous" it is to predict class j is the average cost of
        predicting j:

            r_j = mean_i C_{i,j}

        We return scores ``f_base = -r`` so that low-risk classes have higher
        score (softmin).

        Parameters
        ----------
        C:
            Cost matrix of shape (K,K) or batch (B,K,K).

        Returns
        -------
        Tensor
            - shape (K,) if C is (K,K)
            - shape (B,K) if C is (B,K,K)
        """
        if C.ndim == 2:
            # mean over true class i (rows)
            r = C.mean(dim=0)  # (K,)
            return -r
        if C.ndim == 3:
            r = C.mean(dim=1)  # (B,K)
            return -r
        raise ValueError("C must have shape (K,K) or (B,K,K).")

    # -------------------------------------------------------------------------
    # Core conjugate computation
    # -------------------------------------------------------------------------

    def _conjugate_term(self, scores: Tensor, C: Tensor, eps: Tensor) -> Tensor:
        """
        Compute the Fenchel conjugate term Ω*_{C,ε}(scores) per example.

        Parameters
        ----------
        scores:
            Tensor of shape (B,K).
        C:
            Tensor of shape (B,K,K).
        eps:
            Tensor of shape (B,) (temperature).

        Returns
        -------
        Tensor
            Conjugate term of shape (B,).
        """
        B, K = scores.shape

        # Split scores symmetrically to maintain gauge/shift invariance.
        f_i = (0.5 * scores).unsqueeze(2)  # (B,K,1)
        f_j = (0.5 * scores).unsqueeze(1)  # (B,1,K)

        exponent = -(f_i + f_j + C) / eps.view(-1, 1, 1)  # (B,K,K)

        # Stabilize exp by subtracting max.
        shift = exponent.amax(dim=(1, 2), keepdim=True)  # (B,1,1)
        logM = exponent - shift
        M = torch.exp(logM)

        # Solve inner problem (NO gradient through solver).
        with torch.no_grad():
            alpha = _solve_qp_on_simplex(M, n_iter=self.solver_iter)  # (B,K)

        # Compute log(αᵀ M α) robustly in log-space to avoid underflow.
        # α may contain exact zeros due to the LMO, so we mask log(0) as -inf.
        neginf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
        loga = torch.where(alpha > 0, torch.log(alpha), neginf)  # (B,K)

        term = loga.unsqueeze(2) + loga.unsqueeze(1) + logM  # (B,K,K)
        logval = torch.logsumexp(term.view(B, K * K), dim=1)  # (B,)

        conj = -eps * (logval + shift.squeeze())  # (B,)
        return conj

    def _raw_loss_per_example(self, scores: Tensor, targets: Tensor, C: Tensor) -> Tensor:
        """
        Compute raw CACIS loss per example.

        Parameters
        ----------
        scores:
            Tensor (B,K) of model scores.
        targets:
            Tensor (B,) of integer class labels.
        C:
            Tensor (B,K,K) of cost matrices.

        Returns
        -------
        Tensor
            Raw loss vector of shape (B,).
        """
        if scores.ndim != 2:
            raise ValueError("scores must have shape (B,K).")
        if targets.ndim != 1:
            raise ValueError("targets must have shape (B,).")
        if C.ndim != 3:
            raise ValueError("C must have shape (B,K,K).")

        B, K = scores.shape
        if targets.shape[0] != B:
            raise ValueError("targets and scores must agree on batch size.")
        if C.shape[0] != B or C.shape[1] != K or C.shape[2] != K:
            raise ValueError("C must have shape (B,K,K) matching scores.")

        eps = self._compute_epsilon(C)  # (B,)
        conj = self._conjugate_term(scores, C, eps)  # (B,)

        f_y = scores.gather(1, targets.view(-1, 1)).squeeze(1)  # (B,)
        return conj - f_y

    # -------------------------------------------------------------------------
    # Baseline loss (per example)
    # -------------------------------------------------------------------------

    def _baseline_loss_per_example(self, targets: Tensor, C: Tensor) -> Tensor:
        """
        Compute baseline loss per example.

        Parameters
        ----------
        targets:
            Tensor (B,) of integer class labels.
        C:
            Tensor (B,K,K) of cost matrices.

        Returns
        -------
        Tensor
            Baseline raw loss vector (B,).

        Notes
        -----
        This uses the cost-aware uninformed baseline scores derived from C.
        """
        scores_base = self.baseline_scores(C)  # (B,K)
        return self._raw_loss_per_example(scores_base, targets, C)

    def _maybe_cached_baseline_for_constant_C(self, targets: Tensor, C2: Tensor) -> Optional[Tensor]:
        """
        Fast path for constant cost matrices (shape (K,K)).

        If C is constant across the batch, the conjugate term for baseline scores
        is constant as well, so we can compute it once and reuse it across calls.

        Returns
        -------
        Optional[Tensor]
            Baseline loss vector of shape (B,) if cache path is used, else None.
        """
        if C2.ndim != 2:
            return None

        # Cache key: tensor identity + device/dtype + epsilon settings + solver iters
        cache_ok = (
            self._cache_C_id == id(C2)
            and self._cache_device == C2.device
            and self._cache_dtype == C2.dtype
            and self._cache_solver_iter == self.solver_iter
        )

        # Compute epsilon scalar for constant C (depends on epsilon_mode/settings).
        eps_scalar_t = self._compute_epsilon(C2)  # scalar tensor
        eps_scalar = float(eps_scalar_t.item())
        cache_ok = cache_ok and (self._cache_eps == eps_scalar)

        if not cache_ok:
            # Rebuild cache
            scores_base = self.baseline_scores(C2)  # (K,)

            # Compute conjugate term once (batch size 1).
            C1 = C2.unsqueeze(0)
            scores1 = scores_base.unsqueeze(0)

            # Any target index works; we choose 0 and then add back f_y.
            targets1 = torch.zeros((1,), device=C2.device, dtype=torch.long)

            with torch.no_grad():
                raw1 = self._raw_loss_per_example(scores1, targets1, C1)  # (1,)
                conj_base = (raw1 + scores_base[0]).squeeze(0).detach()

            self._cache_C_id = id(C2)
            self._cache_device = C2.device
            self._cache_dtype = C2.dtype
            self._cache_eps = eps_scalar
            self._cache_solver_iter = self.solver_iter
            self._cache_scores_base = scores_base.detach()
            self._cache_conj_base = conj_base

        assert self._cache_scores_base is not None
        assert self._cache_conj_base is not None

        # base_i = conj_base - scores_base[y_i]
        return self._cache_conj_base - self._cache_scores_base.gather(0, targets)

    # -------------------------------------------------------------------------
    # Main forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        C: Optional[Tensor] = None,
    ) -> CACISLossOutput:
        """
        Compute CACIS raw loss (for backprop) and normalized loss (for logging).

        Parameters
        ----------
        scores:
            Tensor of shape (B,K). These are model outputs (logits/scores).
        targets:
            Tensor of shape (B,) with integer class indices.
        C:
            Cost matrix, either:
            - shape (K,K) for a global cost matrix, or
            - shape (B,K,K) for example-dependent costs.
            If None, defaults to uniform misclassification cost (1 - I).

        Returns
        -------
        CACISLossOutput
            A named tuple ``(loss, loss_norm)`` where:
            - ``loss`` is a scalar torch tensor (mean raw CACIS loss).
            - ``loss_norm`` is a Python float (mean baseline-normalized CACIS).

        Notes
        -----
        - Backpropagate **only** through ``loss``.
        - ``loss_norm`` is computed from detached tensors and is not differentiable.
        - Normalization uses **mean of ratios**: mean_i raw_i / base_i.
        """
        if scores.ndim != 2:
            raise ValueError("scores must have shape (B,K).")
        if targets.ndim != 1:
            raise ValueError("targets must have shape (B,).")

        B, K = scores.shape
        device = scores.device
        dtype = scores.dtype

        # Default cost: 1 - I
        if C is None:
            C = torch.ones((K, K), device=device, dtype=dtype)
            C.fill_diagonal_(0.0)

        # Keep a reference to detect constant-C case.
        C_is_constant = (C.ndim == 2)

        # Expand to batch shape (B,K,K) for raw computation.
        if C.ndim == 2:
            Cb = C.unsqueeze(0).expand(B, -1, -1)
        elif C.ndim == 3:
            if C.shape[0] != B or C.shape[1] != K or C.shape[2] != K:
                raise ValueError("C must have shape (B,K,K) matching scores.")
            Cb = C
        else:
            raise ValueError("C must have shape (K,K) or (B,K,K).")

        # Raw loss (per example), differentiable w.r.t. scores (implicit gradient).
        raw_vec = self._raw_loss_per_example(scores, targets, Cb)  # (B,)
        loss = raw_vec.mean()

        # Normalized loss: detach everything (pure reporting).
        with torch.no_grad():
            if C_is_constant:
                base_vec = self._maybe_cached_baseline_for_constant_C(targets, C)  # (B,)
                if base_vec is None:
                    base_vec = self._baseline_loss_per_example(targets, Cb)
            else:
                base_vec = self._baseline_loss_per_example(targets, Cb)

            tiny = torch.finfo(dtype).tiny
            norm_vec = raw_vec.detach() / base_vec.clamp_min(tiny)
            loss_norm = float(norm_vec.mean().item())

        return CACISLossOutput(loss=loss, loss_norm=loss_norm)
