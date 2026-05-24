"""
utils/metrics.py — Evaluation metrics for HW4.
"""

import numpy as np
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
)


# ─────────────────────────────────────────────────────────────────────────────
# Regression (Parts b, c)
# ─────────────────────────────────────────────────────────────────────────────

def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.nn.functional.mse_loss(preds, targets).item()


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean(torch.abs(preds - targets)).item()


def directional_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of samples where predicted and actual return have the same sign."""
    correct = ((preds > 0) == (targets > 0)).float()
    return correct.mean().item()


def per_horizon_mse(preds: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """MSE for each forecast horizon d = 1 … D independently."""
    return np.array(
        [torch.nn.functional.mse_loss(preds[:, i], targets[:, i]).item()
         for i in range(preds.shape[1])]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Classification (Part d)
# ─────────────────────────────────────────────────────────────────────────────

def binary_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    true  = labels.cpu().numpy().astype(int)

    metrics = {
        "accuracy":  float(np.mean(preds == true)),
        "precision": precision_score(true, preds, zero_division=0),
        "recall":    recall_score(true, preds, zero_division=0),
        "f1":        f1_score(true, preds, zero_division=0),
    }
    if len(np.unique(true)) > 1:
        metrics["auc"] = roc_auc_score(true, probs)
    else:
        metrics["auc"] = float("nan")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Communication system (Part 2)
# ─────────────────────────────────────────────────────────────────────────────

def symbol_error_rate(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits  : (B, 4, alphabet_size)
    targets : (B, 4)   integer
    """
    preds = logits.argmax(dim=-1)              # (B, 4)
    wrong = (preds != targets).float()
    return wrong.mean().item()


def message_error_rate(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fraction of ENTIRE messages decoded incorrectly."""
    preds = logits.argmax(dim=-1)              # (B, 4)
    wrong_msg = (preds != targets).any(dim=1).float()
    return wrong_msg.mean().item()
