"""
CACIS demo: Fraud detection (IEEE-CIS)
=====================================

This script demonstrates CACIS on a realistic fraud detection problem using:
- IEEE-CIS Fraud Detection tabular features
- Example-dependent *binary* cost matrices per transaction
- CACIS loss (Fenchel–Young, cost-aware)

Interpretation of reported CACIS values
---------------------------------------
We report the *baseline-normalized* CACIS loss:
    normalized CACIS = 0  -> perfect decision
    normalized CACIS = 1  -> cost-aware random (uninformed) guessing
    normalized CACIS < 1  -> better than uninformed baseline

Costs
-----
We use a simple, defensible economic model:

- False positive (flagging a legitimate transaction):
    c_FP = review_minutes/60 * hourly_wage
- False negative (missing a fraud):
    c_FN,i = FN_SCALE * TransactionAmt_i

By default:
- c_FP = 5 euros
- FN_SCALE = 1.0

Data split
----------
We train on:
    ieee-fraud-detection/train_transaction.csv
and we *score*:
    ieee-fraud-detection/test_transaction.csv

Important: Kaggle-style test sets do not contain labels, so we cannot compute
AUC or test loss there. We therefore plot training convergence only.

Run
---
    PYTHONPATH=. python examples/fraud_detection.py --epochs 5

Dependencies
------------
    pip install torch pandas numpy matplotlib scikit-learn tqdm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cacis.nn.loss import CACISLoss
from utils import TrainingState, get_device, plot_loss_trajectory, setup_logging

import os

# ============================================================
# Constants & configuration
# ============================================================


# Economic parameters (edit for your context)
FP_COST_EURO = 5.0   # cost of manual review (false positive)
FN_SCALE = 1.0       # multiplier on TransactionAmt (false negative)


# ============================================================
# Dataset
# ============================================================

class FraudTrainDataset(Dataset):
    """
    Training dataset for fraud detection with example-dependent costs.

    Each sample yields:
    - x : float32 features, shape (D,)
    - y : int64 label in {0,1}
    - C : float32 cost matrix, shape (2,2)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df["isFraud"].values.astype(np.int64)

        # RAW amount (euros) for the cost. Do NOT transform for costs.
        amount = df["TransactionAmt"].values.astype(np.float32)

        # Build per-example costs:
        #   C = [[0, c_FP],
        #        [c_FN, 0]]
        self.C = np.zeros((len(self.y), 2, 2), dtype=np.float32)
        self.C[:, 0, 1] = FP_COST_EURO
        self.C[:, 1, 0] = FN_SCALE * amount

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.tensor(self.y[idx]),
            torch.from_numpy(self.C[idx]),
        )


class FraudTestDataset(Dataset):
    """
    Kaggle-style test dataset with no labels (used only for scoring).

    Each sample yields:
    - x : float32 features, shape (D,)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        self.X = df[feature_cols].values.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx])


# ============================================================
# Model
# ============================================================

class LogisticModel(nn.Module):
    """Simple logistic regression baseline (2-class)."""
    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CACIS fraud detection demo (IEEE-CIS)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--quick", action="store_true", help="Quick run (few batches)")
    parser.add_argument("--out", type=str, default="fraud_output", help="Output directory")
    args = parser.parse_args()

    setup_logging()
    device = get_device()
    logging.info("Using device: %s", device)

    OUTPUT_DIR = args.out   
    if not Path(OUTPUT_DIR).exists():
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------------
    # Load train / test (official Kaggle split)
    # --------------------------------------------------------
    data_folder = "ieee-fraud-detection"
    data_files = ["train_transaction.csv", "test_transaction.csv"]
    data_files = [os.path.join(data_folder, f) for f in data_files]
    dwnld_msg = "rm -rf ieee-fraud-detection.zip ieee-fraud-detection || true\nmkdir ieee-fraud-detection\nwget -c http://deraison.ai/ai/ieee-fraud-detection.zip\nunzip ieee-fraud-detection.zip -d ieee-fraud-detection"
    for f in data_files:
        if not Path(f).exists():
            logging.error("Missing file: %s\nRun:\n%s", f, dwnld_msg)
            return

    train_df = pd.read_csv(data_files[0])
    test_df = pd.read_csv(data_files[1])

    # Add a *feature* column with log amount, but keep TransactionAmt raw.
    # Feature side: you MAY add log(amt) but must keep amt itself raw.
    for df in (train_df, test_df):
        df["TransactionAmt_log1p"] = np.log1p(df["TransactionAmt"].astype(float))

    # Exclude identifier columns from features (keep for output if present).
    # TransactionID is an ID, not a predictive signal.
    drop_cols = {"isFraud", "TransactionID"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # One-hot encode categoricals on train, align test to train columns.
    categorical_cols = train_df[feature_cols].select_dtypes(include=["object"]).columns
    train_df = pd.get_dummies(train_df, columns=categorical_cols, dummy_na=True)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, dummy_na=True)
    train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    train_df[feature_cols] = train_df[feature_cols].fillna(0.0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0.0)

    logging.info("Final feature dimension: %d", len(feature_cols))

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    train_ds = FraudTrainDataset(train_df, feature_cols)
    test_ds = FraudTestDataset(test_df, feature_cols)

    batch_size = args.batch_size
    state = TrainingState(batch_size=batch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # --------------------------------------------------------
    # Model / loss / optimizer
    # --------------------------------------------------------
    model = LogisticModel(len(feature_cols)).to(device)

    # CACIS loss: returns (loss_tensor, loss_norm_float) in one call.
    cacis_loss = CACISLoss(epsilon_mode="offdiag_max", solver_iter=30)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    # --------------------------------------------------------
    # Training loop (convergence on normalized CACIS)
    # --------------------------------------------------------
    logging.info("Starting training...")

    for epoch in range(args.epochs):
        model.train()
        epoch_norm_sum = 0.0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", total=len(train_loader))
        for step, (x, y, C) in enumerate(pbar):
            if args.quick and step >= 5:
                break

            x, y, C = x.to(device), y.to(device), C.to(device)

            optimizer.zero_grad()
            scores = model(x)

            loss, loss_norm = cacis_loss(scores, y, C=C)

            loss.backward()
            optimizer.step()

            state.training_loss_history.append(loss_norm)
            state.current_iter += 1

            epoch_norm_sum += loss_norm
            epoch_steps += 1

            pbar.set_postfix({"loss_norm": f"{loss_norm:.4f}"})

        state.epoch_iterations.append(state.current_iter)
        logging.info("Epoch %02d | Mean normalized CACIS: %.4f", epoch + 1, epoch_norm_sum / max(1, epoch_steps))

        plot_loss_trajectory(
            state,
            out_path=os.path.join(OUTPUT_DIR, "fraud_loss_trajectory.png"),
            title="CACIS Normalized Loss — IEEE-CIS Fraud Detection",
        )

    logging.info("Training finished.")

    # --------------------------------------------------------
    # Score test set and save predictions (Kaggle-style)
    # --------------------------------------------------------
    model.eval()
    probs: List[float] = []
    with torch.no_grad():
        for x in tqdm(test_loader, desc="Scoring test"):
            x = x.to(device)
            s = model(x)
            p = torch.softmax(s, dim=1)[:, 1]
            probs.extend(p.cpu().numpy().tolist())

    out_csv = os.path.join(OUTPUT_DIR, "test_predictions.csv")
    if "TransactionID" in test_df.columns:
        submission = pd.DataFrame({"TransactionID": test_df["TransactionID"].values, "isFraud": probs})
    else:
        submission = pd.DataFrame({"isFraud": probs})

    submission.to_csv(out_csv, index=False)
    logging.info("Saved test predictions to: %s", out_csv)

    logging.info("Demo finished successfully.")


if __name__ == "__main__":
    main()
