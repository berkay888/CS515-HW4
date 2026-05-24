"""
train.py — Training loops for all HW4 experiments.

Usage (called from main.py, or directly):
  python train.py --experiment return_lstm
"""

import os
import copy
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, List

CKPT_DIR = "results/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Generic regression trainer (Parts b, c)
# ─────────────────────────────────────────────────────────────────────────────

def train_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    ckpt_name: str,
) -> Tuple[List[float], List[float]]:
    """
    Train a regression model (MSE loss) with early stopping.
    Returns (train_losses, val_losses) per epoch.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)

        # ── Validate ──
        val_loss = evaluate_regression(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")

        # ── Early stopping ──
        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total += criterion(pred, y).item() * X.size(0)
            n += X.size(0)
    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# Classification trainer (Part d)
# ─────────────────────────────────────────────────────────────────────────────

def train_classification(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    ckpt_name: str,
    pos_weight: Optional[torch.Tensor] = None,
) -> Tuple[List[float], List[float]]:
    """Binary classification with BCEWithLogitsLoss + early stopping."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pw = pos_weight.to(device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val   = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X).squeeze(-1)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)

        val_loss = evaluate_classification(model, val_loader, device, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | train BCE {train_loss:.4f} | val BCE {val_loss:.4f}")

        if val_loss < best_val - 1e-7:
            best_val   = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses


def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X).squeeze(-1)
            total += criterion(logits, y).item() * X.size(0)
            n += X.size(0)
    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# Communication system trainer (Part 2)
# ─────────────────────────────────────────────────────────────────────────────

def train_comm(
    model: nn.Module,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device: torch.device,
    alphabet_size: int = 8,
    seq_len: int = 4,
    ckpt_name: str = "comm_system",
):
    """
    Train the interactive communication system.
    Messages are randomly generated on the fly — no fixed dataset needed.
    """
    from utils.metrics import symbol_error_rate, message_error_rate

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    val_size   = 2048
    # Fixed validation set
    val_msgs = torch.randint(0, alphabet_size, (val_size, seq_len), device=device)

    best_mer   = float("inf")
    best_state = None
    no_improve = 0
    train_losses, val_sers, val_mers = [], [], []

    steps_per_epoch = 200   # mini-batches per epoch

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for _ in range(steps_per_epoch):
            msgs = torch.randint(0, alphabet_size, (batch_size, seq_len), device=device)
            optimizer.zero_grad()
            logits = model(msgs)   # (B, 4, A)
            # logits: (B*4, A), targets: (B*4,)
            loss = criterion(logits.view(-1, alphabet_size), msgs.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= steps_per_epoch
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_msgs)
            ser = symbol_error_rate(val_logits, val_msgs)
            mer = message_error_rate(val_logits, val_msgs)

        train_losses.append(ep_loss)
        val_sers.append(ser)
        val_mers.append(mer)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | loss {ep_loss:.4f} | SER {ser:.4f} | MER {mer:.4f}")

        if mer < best_mer - 1e-5:
            best_mer   = mer
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, os.path.join(CKPT_DIR, f"{ckpt_name}_best.pth"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return train_losses, val_sers, val_mers
