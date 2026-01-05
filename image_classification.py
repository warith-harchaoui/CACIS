"""
CACIS demo: Image classification (ResNet + CIFAR-10 + fastText costs)
====================================================================

This script demonstrates CACIS on CIFAR-10 using:
- ResNet18 (ImageNet pretrained)
- CACIS loss (Fenchel–Young, cost-aware)
- fastText pretrained embeddings to build a semantic cost matrix

Additionally:
- Per-epoch fine confusion matrix (10x10)
- Per-epoch grouped confusion matrix (Animals vs Vehicles)
- A convergence curve of the *normalized* CACIS (0 perfect, 1 baseline)

Interpretation of reported CACIS values
---------------------------------------
We report the *baseline-normalized* CACIS loss:
    normalized CACIS = 0  -> perfect prediction
    normalized CACIS = 1  -> cost-aware random (uninformed) guessing
    normalized CACIS < 1  -> better than uninformed baseline

Run
---
    PYTHONPATH=. python examples/image_classification.py --epochs 10

Dependencies
------------
    pip install torch torchvision fasttext matplotlib seaborn scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from pathlib import Path
from typing import List

import fasttext
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from cacis.nn.loss import CACISLoss
from utils import TrainingState, get_device, plot_loss_trajectory, setup_logging


# ============================================================
# Constants & Configuration
# ============================================================

FASTTEXT_DIR = Path("./data/fasttext")
FASTTEXT_MODEL_PATH = FASTTEXT_DIR / "cc.en.300.bin"

# NOTE: you can replace this URL with the official fastText URL if preferred.
FASTTEXT_URL = "https://deraison.ai/ai/cc.en.300.bin"

IMAGES_DIR = Path("images")

ANIMALS = {"bird", "cat", "deer", "dog", "frog", "horse"}
VEHICLES = {"airplane", "automobile", "ship", "truck"}


# ============================================================
# fastText & Semantic Costs
# ============================================================

def load_fasttext_model() -> fasttext.FastText:
    """
    Download (if needed) and load fastText pretrained vectors.

    Returns
    -------
    fasttext.FastText
        Loaded fastText model.
    """
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)

    if not FASTTEXT_MODEL_PATH.exists():
        logging.info("Downloading fastText model: %s", FASTTEXT_MODEL_PATH)
        urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_MODEL_PATH)

    logging.info("Loading fastText model (this can take a few seconds)...")
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


def build_semantic_cost_matrix(class_names: List[str], scale: float = 5.0) -> torch.Tensor:
    """
    Build a (K,K) semantic cost matrix using fastText embeddings.

    C_ij = scale * (1 - cosine_similarity(emb(word_i), emb(word_j)))
    C_ii = 0

    Parameters
    ----------
    class_names:
        List of class names (e.g., CIFAR-10 class names).
    scale:
        Multiplicative scale for semantic distances.

    Returns
    -------
    torch.Tensor
        Cost matrix of shape (K,K), float32.
    """
    ft = load_fasttext_model()

    vectors = []
    for name in class_names:
        v = ft.get_word_vector(name.lower())
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors.append(v)

    V = torch.tensor(np.stack(vectors), dtype=torch.float32)  # (K,D)
    S = V @ V.T  # cosine similarity
    C = 1.0 - S
    C = torch.clamp(C, min=0.0)
    C.fill_diagonal_(0.0)

    return scale * C


def grouped_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> np.ndarray:
    """
    Reduce a fine confusion matrix into 2 groups: Animals vs Vehicles.

    Parameters
    ----------
    cm:
        Confusion matrix of shape (K,K).
    class_names:
        Class names of length K.

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (2,2).
    """
    group_map = {i: (0 if name in ANIMALS else 1) for i, name in enumerate(class_names)}
    cm_group = np.zeros((2, 2), dtype=int)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            cm_group[group_map[i], group_map[j]] += cm[i, j]
    return cm_group


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CACIS image classification demo (CIFAR-10)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick run (few batches)")
    args = parser.parse_args()

    setup_logging()
    device = get_device()
    logging.info("Using device: %s", device)

    IMAGES_DIR.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    class_names = trainset.classes
    num_classes = len(class_names)
    logging.info("Classes: %s", class_names)

    # --------------------------------------------------------
    # Cost matrix
    # --------------------------------------------------------
    cost_matrix = build_semantic_cost_matrix(class_names, scale=5.0).to(device)
    mean_cost = cost_matrix[cost_matrix > 0].mean().item()
    logging.info("Mean off-diagonal semantic cost: %.4f", mean_cost)

    # Visualize cost matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cost_matrix.detach().cpu().numpy(), cmap="viridis",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Semantic cost matrix (fastText)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "cost_matrix.png")
    plt.close()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # --------------------------------------------------------
    # Loss & optimizer
    # --------------------------------------------------------
    cacis_loss = CACISLoss(epsilon_mode="offdiag_max", solver_iter=30)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    state = TrainingState(batch_size=args.batch_size)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    logging.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        epoch_norm_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
        for step, (x, y) in enumerate(pbar):
            if args.quick and step >= 5:
                break

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            scores = model(x)

            loss, loss_norm = cacis_loss(scores, y, C=cost_matrix)

            loss.backward()
            optimizer.step()

            state.training_loss_history.append(loss_norm)
            state.current_iter += 1

            epoch_norm_sum += loss_norm
            epoch_steps += 1

            pbar.set_postfix({"loss_norm": f"{loss_norm:.4f}"})

        logging.info("Epoch %02d | Mean normalized CACIS: %.4f", epoch + 1, epoch_norm_sum / max(1, epoch_steps))
        state.epoch_iterations.append(state.current_iter)

        # ----------------------------------------------------
        # Evaluation (normalized CACIS + confusion matrices)
        # ----------------------------------------------------
        model.eval()
        y_true: List[int] = []
        y_pred: List[int] = []
        test_norm_sum = 0.0
        test_steps = 0

        with torch.no_grad():
            for step, (x, y) in enumerate(test_loader):
                if args.quick and step >= 5:
                    break

                x, y = x.to(device), y.to(device)
                scores = model(x)

                _, loss_norm = cacis_loss(scores, y, C=cost_matrix)
                test_norm_sum += loss_norm
                test_steps += 1

                preds = scores.argmax(dim=1)
                y_true.extend(y.cpu().numpy().tolist())
                y_pred.extend(preds.cpu().numpy().tolist())

        test_norm = test_norm_sum / max(1, test_steps)
        state.test_loss_history.append(test_norm)

        logging.info("Epoch %02d | Test normalized CACIS: %.4f", epoch + 1, test_norm)

        # Plot loss trajectory
        plot_loss_trajectory(
            state,
            out_path=IMAGES_DIR / "loss_trajectory.png",
            title="Optimization Trajectory for CIFAR-10 + ResNet18 with fastText semantic costs",
        )

        # Confusion matrices
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f"Confusion matrix (epoch {epoch+1})")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"confusion_matrix_{epoch+1}.png")
        plt.close()

        cm_group = grouped_confusion_matrix(cm, class_names)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm_group, annot=True, fmt="d", cmap="viridis",
                    xticklabels=["Animal", "Vehicle"], yticklabels=["Animal", "Vehicle"])
        plt.title(f"Grouped confusion (epoch {epoch+1})")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / f"confusion_matrix_grouped_{epoch+1}.png")
        plt.close()

    logging.info("Training finished.")
    logging.info("Demo finished successfully.")


if __name__ == "__main__":
    main()
