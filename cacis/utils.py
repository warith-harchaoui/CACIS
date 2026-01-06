"""
utils
==============

Small, reusable utilities shared across CACIS example scripts.

This module is intentionally lightweight and focused on:
- logging configuration
- device selection (CUDA / MPS / CPU)
- a small TrainingState container for plotting convergence curves
- plotting a CACIS loss trajectory with the "0 perfect / 1 baseline" semantics

The goal is for example scripts to remain short and readable while sharing
consistent behavior across domains (vision, tabular, etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


PathLike = Union[str, Path]


@dataclass
class TrainingState:
    """
    Hold training history for CACIS demos.

    Attributes
    ----------
    training_loss_history:
        List of *normalized* CACIS values recorded at each optimization step
        (one value per mini-batch).
    test_loss_history:
        Optional list of *normalized* CACIS values recorded at epoch boundaries.
        For datasets without labels (e.g., Kaggle test sets), keep this empty.
    epoch_iterations:
        Iteration indices at epoch boundaries, used to draw vertical lines.
        Convention: starts with [0].
    current_iter:
        Current iteration counter (increments once per mini-batch).
    batch_size:
        Batch size, used only for plot annotation.
    """
    training_loss_history: List[float] = field(default_factory=list)
    test_loss_history: List[float] = field(default_factory=list)
    epoch_iterations: List[int] = field(default_factory=lambda: [0])
    current_iter: int = 0
    batch_size: int = 0


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure basic logging for example scripts.

    Parameters
    ----------
    level:
        Logging level (e.g., logging.INFO).
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_device() -> torch.device:
    """
    Return the best available PyTorch device.

    Returns
    -------
    torch.device
        "cuda" if available, else "mps" if available, else "cpu".
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def moving_average(input_x: np.ndarray, window_size: int, axis: int = 0) -> np.ndarray:
    """
    Causal moving average for 1D arrays.

    Parameters
    ----------
    input_x:
        Input time series.
    window_size:
        Window size for the moving average. If <= 1, returns x unchanged.
    axis:
        Axis along which to compute the moving average which is the time axis. (default: 0)

    Returns
    -------
    np.ndarray
        Smoothed array (shorter than x when window_size > 1).
    """
    x = np.asarray(input_x)
    if window_size < 0:
        raise ValueError("window_size must be >= 0")

    n = x.shape[axis]
    if n == 0:
        return np.zeros_like(x)

    if n <= window_size + 1:
        m = np.mean(x)
        return np.full_like(x, m)

    L = window_size + 1
    work_dtype = np.result_type(x.dtype, np.float64)

    y = np.cumsum(x, axis=axis, dtype=work_dtype)
    if L < n:
        hi = [slice(None)] * x.ndim
        lo = [slice(None)] * x.ndim
        hi[axis] = slice(L, None)
        lo[axis] = slice(None, -L)
        y[tuple(hi)] -= y[tuple(lo)]

    denom = np.minimum(np.arange(1, n + 1), L).astype(work_dtype)
    shape = [1] * x.ndim
    shape[axis] = n
    y /= denom.reshape(shape)

    return y.astype(x.dtype, copy=False)


def plot_loss_trajectory(
    state: TrainingState,
    *,
    out_path: str = "images/loss_trajectory.png",
    title: str = "CACIS Loss",
    ylabel: str = "Loss",
    ma_window: Optional[int] = 20,
    normalize: bool = True,
    crop: bool = False,
) -> None:
    """
    Plot a CACIS loss trajectory.

    The plot uses the CACIS normalization convention:
    - y = 0: perfect prediction
    - y = 1: cost-aware baseline (uninformed) guessing

    Parameters
    ----------
    state:
        Training state containing recorded losses.
    out_path:
        Where to save the figure.
    title:
        Title of the plot.
    ylabel:
        Y-axis label.
    ma_window:
        Optional moving-average window on training curve. If None, do not plot MA.
    y_limits:
        y-axis limits.

    Notes
    -----
    This function saves directly to disk and closes the figure.
    """

    plt.figure(figsize=(18, 6))

    # Training curve (normalized CACIS, step-wise)
    plt.plot(
        state.training_loss_history,
        label="Training (step-wise)",
        alpha=0.5,
        color = "#007AFF"
    )

    values = set()
    values = values.union(set(list(state.training_loss_history)))




    # Optional moving average
    if ma_window is not None and len(state.training_loss_history) >= ma_window:
        smoothed = moving_average(np.asarray(state.training_loss_history), ma_window)
        plt.plot(smoothed, label=f"Training ({ma_window}-moving-average)", linewidth=3)
        values = values.union(set(smoothed))

    # Optional test/validation curve at epoch boundaries
    if state.test_loss_history:
        plt.plot(
            state.epoch_iterations[1:],
            state.test_loss_history,
            label="Test (per-epoch)",
            linewidth=3,
            color="#AF52DE"
        )
        values = values.union(set(list(state.test_loss_history)))

    # Baseline reference lines
    if normalize:
        plt.axhline(1.0, linestyle="--", label="Cost-aware random guessing (1.0)", color="#FF3B30")

    if crop:
        M = np.quantile(np.array(list(values)), 0.90)
    else:
        M = np.max(np.array(list(values)))
    
    y_limits = (-0.1 * M, 1.2 * M)

    plt.axhline(0.0, linestyle="--", label="Perfect (0.0)", color="#28CD41")

    # Epoch markers
    b = True
    for it in state.epoch_iterations:
        plt.axvline(it, color="gray", linestyle=":", label="Epochs" if b else None)
        b = False

    plt.xlabel(f"Iterations (1 iter = 1 mini-batch = {state.batch_size} examples)")
    plt.ylabel(ylabel)
    plt.title(title + "\n(the lower, the better)")
    plt.ylim(*y_limits)

    if state.epoch_iterations:
        d = 0.1 * max(state.epoch_iterations)
        plt.xlim(-d, max(state.epoch_iterations) + d)

    # Light style
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    o = out_path.replace(".png", ".txt")
    np.savetxt(o, state.training_loss_history)
