"""
utils/visualization.py — Training-curve and result plots for HW4.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)


def _savefig(name: str):
    path = os.path.join(SAVE_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Generic training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses,   label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _savefig(fname)


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) / (c) — per-horizon MSE bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_horizon_mse(
    mse_lstm: np.ndarray,
    mse_gru:  np.ndarray,
    title:    str,
    fname:    str,
    horizons=(1, 2, 3, 4, 5),
):
    x = np.arange(len(horizons))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, mse_lstm, w, label="LSTM", color="#4C72B0")
    ax.bar(x + w / 2, mse_gru,  w, label="GRU",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in horizons])
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _savefig(fname)


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) vs (c) comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_return_vs_rolling(
    mse_return_lstm, mse_rolling_lstm,
    mse_return_gru,  mse_rolling_gru,
    horizons=(1, 2, 3, 4, 5),
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (m_ret, m_rol, name) in zip(
        axes,
        [(mse_return_lstm, mse_rolling_lstm, "LSTM"),
         (mse_return_gru,  mse_rolling_gru,  "GRU")],
    ):
        x = np.arange(len(horizons))
        w = 0.35
        ax.bar(x - w / 2, m_ret, w, label="Exact Return",   color="#4C72B0")
        ax.bar(x + w / 2, m_rol, w, label="Rolling Return",  color="#55A868")
        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in horizons])
        ax.set_ylabel("MSE")
        ax.set_title(f"{name}: Exact vs Rolling Return MSE")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _savefig("return_vs_rolling_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Part (d) — confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pass", "Buy"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Pass", "Buy"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    _savefig(fname)


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — SER / MER learning curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_comm_curves(train_losses, val_sers, val_mers, fname="comm_learning_curves.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, color="#4C72B0")
    ax1.set_title("Training Loss (Cross-Entropy)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)

    ax2.plot(val_sers,  label="Symbol Error Rate")
    ax2.plot(val_mers,  label="Message Error Rate")
    ax2.set_title("Validation Error Rates")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Rate")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fname)


def plot_snr_vs_error(snr_db_list, ser_list, mer_list, fname="comm_snr_curve.png"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(snr_db_list, ser_list, "o-", label="SER")
    ax.semilogy(snr_db_list, mer_list, "s--", label="MER")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Error Rate (log scale)")
    ax.set_title("Error Rate vs SNR — Interactive AWGN System")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    _savefig(fname)
