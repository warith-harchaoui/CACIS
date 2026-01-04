"""
CACIS demo: ResNet + CIFAR-10 + fastText semantic costs
======================================================

This script demonstrates CACIS on CIFAR-10 using:
- ResNet18 (ImageNet pretrained)
- CACIS loss (Fenchel–Young, cost-aware)
- fastText pretrained embeddings for semantic costs

Additionally:
- Per-epoch fine confusion matrix (10x10)
- Per-epoch grouped confusion matrix (Animals vs Vehicles)

Interpretation of reported CACIS values
---------------------------------------
We report the *baseline-normalized* CACIS loss:

    normalized CACIS = 0  -> perfect prediction
    normalized CACIS = 1  -> cost-aware random (uninformed) guessing
    normalized CACIS < 1  -> better than random
    normalized CACIS > 1  -> worse than random

This is the direct analogue of cross-entropy expressed in base K.

fastText model (auto-downloaded on first run):
    cc.en.300.bin

Run:
    PYTHONPATH=. python resnet_cifar10_fasttext_example.py
"""

import os
from pathlib import Path
import urllib.request
import logging

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.models import ResNet18_Weights

import numpy as np
import fasttext

from cacis.nn.loss import CACISLoss

from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


# ============================================================
# Logging
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ============================================================
# fastText utilities
# ============================================================

FASTTEXT_DIR = Path("./data/fasttext")
FASTTEXT_MODEL_PATH = FASTTEXT_DIR / "cc.en.300.bin"
FASTTEXT_URL = "https://deraison.ai/ai/cc.en.300.bin"


def load_fasttext_model():
    """
    Download (if needed) and load fastText pretrained vectors.
    """
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)

    if not FASTTEXT_MODEL_PATH.exists():
        logging.info("Downloading fastText model (cc.en.300.bin)...")
        urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_MODEL_PATH)
        logging.info("fastText model stored at: %s", FASTTEXT_MODEL_PATH)

    logging.info("Loading fastText model...")
    return fasttext.load_model(str(FASTTEXT_MODEL_PATH))


# ============================================================
# Semantic cost matrix (fastText)
# ============================================================

def build_semantic_cost_matrix(class_names, scale=5.0):
    """
    Build a (K, K) semantic cost matrix using fastText embeddings.

    C_ij = scale * (1 - cosine_similarity(class_i, class_j))
    C_ii = 0

    The scale factor controls the physical unit of semantic distance.
    """
    logging.info(
        "Building semantic cost matrix for %d classes using fastText...",
        len(class_names),
    )

    ft = load_fasttext_model()

    vectors = []
    for name in class_names:
        v = ft.get_word_vector(name.lower())
        v = v / (np.linalg.norm(v) + 1e-9)
        vectors.append(v)

    V = torch.tensor(np.stack(vectors), dtype=torch.float32)
    S = V @ V.T
    C = 1.0 - S
    C = torch.clamp(C, min=0.0)
    C.fill_diagonal_(0.0)

    return scale * C


# ============================================================
# Grouped confusion matrix (Animals vs Vehicles)
# ============================================================

def grouped_confusion_matrix(cm, class_names):
    """
    Aggregate a 10x10 CIFAR-10 confusion matrix into a 2x2 matrix:
    Animals vs Vehicles.
    """
    ANIMALS = {"bird", "cat", "deer", "dog", "frog", "horse"}
    VEHICLES = {"airplane", "automobile", "ship", "truck"}

    group_of = {}
    for idx, name in enumerate(class_names):
        if name in ANIMALS:
            group_of[idx] = 0  # Animal
        elif name in VEHICLES:
            group_of[idx] = 1  # Vehicle
        else:
            raise ValueError(f"Unknown CIFAR-10 class: {name}")

    cm_group = np.zeros((2, 2), dtype=int)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            cm_group[group_of[i], group_of[j]] += cm[i, j]

    return cm_group


# ============================================================
# Main
# ============================================================

def main():

    # -------- Device --------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logging.info("Using device: %s", device)

    # -------- Dataset --------
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    testset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    batch_size = 64
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    class_names = trainset.classes
    num_classes = len(class_names)

    logging.info("Classes: %s", class_names)

    # -------- Cost matrix --------
    cost_matrix = build_semantic_cost_matrix(class_names, scale=5.0).to(device)
    mean_cost = cost_matrix[cost_matrix > 0].mean().item()
    logging.info("Mean off-diagonal semantic cost: %.4f", mean_cost)

    # -------- Model --------
    # model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet18()

    for p in model.parameters():
        p.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    # -------- Loss & Optimizer --------
    # NOTE:
    # - raw CACIS is used for optimization
    # - normalized CACIS is used only for interpretation / logging
    cacis_loss = CACISLoss(
        epsilon_mode="offdiag_max",
        solver_iter=30,
    )

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # -------- Output dirs --------
    os.makedirs("images", exist_ok=True)

    # -------- Cost matrix visualization --------
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cost_matrix.detach().cpu().numpy(),
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Semantic cost matrix (fastText)")
    plt.tight_layout()
    plt.savefig("images/cost_matrix.png")
    plt.close()

    # ========================================================
    # Training loop
    # ========================================================

    epochs = 10
    logging.info("Starting training...")

    ell = []

    for epoch in range(epochs):
        model.train()
        total_raw = 0.0
        total_norm = 0.0

        for x, y in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            total=len(train_loader),
        ):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            scores = model(x)

            # Raw loss for optimization
            loss = cacis_loss(scores, y, C=cost_matrix, normalized=False)
            loss.backward()
            optimizer.step()

            # Normalized loss for interpretation (no gradient effect)
            with torch.no_grad():
                loss_norm = cacis_loss(scores, y, C=cost_matrix, normalized=True)
                ell.append(loss_norm.item())

            total_raw += loss.item()
            total_norm += loss_norm.item()

        logging.info(
            "Epoch %02d | CACIS raw: %.4f | CACIS normalized: %.4f",
            epoch + 1,
            total_raw / len(train_loader),
            total_norm / len(train_loader),
        )

        np.savetxt("images/training_normalized_loss.txt", ell)

        # plot training normalized loss
        plt.figure(figsize=(8, 6))
        plt.plot(ell, label="CACIS normalized loss")
        # By construction, the normalized CACIS baseline is 1.0.
        # This baseline is *cost-aware* (depends on C) and reduces to uniform
        # guessing only in the special case C = 1 - I.
        plt.plot(np.arange(len(ell)), np.ones(len(ell)), label="Cost-aware baseline (normalized = 1)")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Training normalized loss")
        plt.savefig("images/training_normalized_loss.png")
        plt.close()

        # ====================================================
        # Evaluation
        # ====================================================

        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc="Evaluation", total=len(test_loader)):
                x, y = x.to(device), y.to(device)
                scores = model(x)
                y_pred.extend(scores.argmax(dim=1).cpu().numpy())
                y_true.extend(y.cpu().numpy())

        # -------- Fine confusion matrix --------
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            annot=True,
            fmt="d",
        )
        plt.title(f"Confusion matrix (epoch {epoch})")
        plt.tight_layout()
        plt.savefig(f"images/confusion_matrix_{epoch}.png")
        plt.close()

        # -------- Grouped confusion matrix --------
        cm_group = grouped_confusion_matrix(cm, class_names)
        plt.figure(figsize=(4, 4))
        sns.heatmap(
            cm_group,
            annot=True,
            fmt="d",
            cmap="viridis",
            xticklabels=["Animal", "Vehicle"],
            yticklabels=["Animal", "Vehicle"],
        )
        plt.title(f"Grouped confusion (epoch {epoch})")
        plt.tight_layout()
        plt.savefig(f"images/confusion_matrix_grouped_{epoch}.png")
        plt.close()

    logging.info("Training finished.")
    logging.info("Demo finished successfully.")


if __name__ == "__main__":
    main()
