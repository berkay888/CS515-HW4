"""
test.py — Load a saved checkpoint and evaluate on the test set.

Examples
--------
# Part b — regression
python test.py --experiment return --model lstm \
    --checkpoint results/checkpoints/return_lstm_best.pth

# Part d — turning point
python test.py --experiment turning_point --model bilstm \
    --checkpoint results/checkpoints/tp_bilstm_best.pth

# Part 2 — communication system
python test.py --experiment comm \
    --checkpoint results/checkpoints/comm_system_best.pth
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from parameters import DataConfig, ReturnConfig, RollingConfig, TurningPointConfig, CommConfig
from models.stock_lstm import StockLSTM
from models.stock_gru  import StockGRU
from models.comm_system import InteractiveCommSystem
from utils.dataset import download_data, build_return_loaders, build_rolling_loaders, build_tp_loaders
from utils.metrics import (
    per_horizon_mse, mae, directional_accuracy,
    binary_metrics, symbol_error_rate, message_error_rate,
)
from utils.visualization import plot_confusion_matrix, plot_snr_vs_error


# ─────────────────────────────────────────────────────────────────────────────

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).cpu()
            all_preds.append(pred)
            all_targets.append(y)
    return torch.cat(all_preds), torch.cat(all_targets)


# ─────────────────────────────────────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────────────────────────────────────

def test_regression(args):
    print(f"\n{'='*60}")
    print(f"  TEST — {args.experiment.upper()} | model: {args.model}")
    print(f"{'='*60}")

    device = _device()
    data_cfg   = DataConfig()
    ret_cfg    = ReturnConfig()
    roll_cfg   = RollingConfig()

    print("Downloading data …")
    all_data = download_data(data_cfg)

    n_features = len(data_cfg.features)
    n_horizons = len(data_cfg.horizons)

    if args.model == "lstm":
        model = StockLSTM(n_features, ret_cfg.hidden_size, ret_cfg.num_layers, n_horizons, ret_cfg.dropout)
    else:
        model = StockGRU(n_features, ret_cfg.hidden_size, ret_cfg.num_layers, n_horizons, ret_cfg.dropout)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    if args.experiment == "return":
        _, _, test_loader, _ = build_return_loaders(all_data, data_cfg, ret_cfg.batch_size)
    else:
        _, _, test_loader = build_rolling_loaders(all_data, data_cfg, roll_cfg, roll_cfg.batch_size)

    preds, targets = _collect(model, test_loader, device)
    h_mse = per_horizon_mse(preds, targets)
    overall_mse = h_mse.mean()
    overall_mae = mae(preds, targets)
    da          = directional_accuracy(preds, targets)

    print(f"\n  Overall MSE : {overall_mse:.6f}")
    print(f"  Overall MAE : {overall_mae:.6f}")
    print(f"  Dir. Acc.   : {da:.4f}")
    print(f"\n  Per-horizon MSE:")
    for d, m in zip(data_cfg.horizons, h_mse):
        print(f"    d={d}: {m:.6f}")


def test_turning_point(args):
    print(f"\n{'='*60}")
    print(f"  TEST — TURNING POINT | model: {args.model}")
    print(f"{'='*60}")

    device  = _device()
    data_cfg = DataConfig()
    tp_cfg   = TurningPointConfig()

    print("Downloading data …")
    all_data = download_data(data_cfg)

    n_features = len(data_cfg.features)
    bidirectional = True

    if "lstm" in args.model:
        model = StockLSTM(n_features, tp_cfg.hidden_size, tp_cfg.num_layers,
                          output_size=1, dropout=tp_cfg.dropout, bidirectional=bidirectional)
    else:
        model = StockGRU(n_features, tp_cfg.hidden_size, tp_cfg.num_layers,
                         output_size=1, dropout=tp_cfg.dropout, bidirectional=bidirectional)

    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    _, _, test_loader, _ = build_tp_loaders(all_data, data_cfg, tp_cfg, tp_cfg.batch_size)
    logits, labels = _collect(model, test_loader, device)
    logits = logits.squeeze(-1)

    metrics = binary_metrics(logits, labels)
    print(f"\n  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  AUC       : {metrics['auc']:.4f}")

    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels.numpy().astype(int), preds)
    print(f"\n  Confusion Matrix:\n{cm}")
    plot_confusion_matrix(cm, f"Turning Point — {args.model.upper()}", f"cm_{args.model}.png")


def test_comm(args):
    print(f"\n{'='*60}")
    print(f"  TEST — INTERACTIVE COMM SYSTEM")
    print(f"{'='*60}")

    device  = _device()
    cfg     = CommConfig()
    model   = InteractiveCommSystem(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate at fixed σ²=0.25
    n_test = 10000
    msgs = torch.randint(0, cfg.alphabet_size, (n_test, cfg.seq_len), device=device)
    with torch.no_grad():
        logits = model(msgs)
    ser = symbol_error_rate(logits, msgs)
    mer = message_error_rate(logits, msgs)
    print(f"\n  @ σ²={cfg.noise_var}")
    print(f"  Symbol Error Rate  : {ser:.4f}")
    print(f"  Message Error Rate : {mer:.4f}")

    # SNR sweep
    print("\n  SNR sweep …")
    snr_db_list = list(range(-5, 16, 2))
    ser_list, mer_list = [], []
    for snr_db in snr_db_list:
        snr_linear = 10 ** (snr_db / 10)
        sigma2     = 1.0 / snr_linear
        old_var    = model.cfg.noise_var
        model.cfg.noise_var = sigma2
        with torch.no_grad():
            logits = model(msgs)
        ser_list.append(symbol_error_rate(logits, msgs))
        mer_list.append(message_error_rate(logits, msgs))
        model.cfg.noise_var = old_var
        print(f"    SNR {snr_db:+3d} dB → SER {ser_list[-1]:.4f} | MER {mer_list[-1]:.4f}")

    plot_snr_vs_error(snr_db_list, ser_list, mer_list)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CS515 HW4 — Test")
    parser.add_argument("--experiment", required=True,
                        choices=["return", "rolling", "turning_point", "comm"])
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "gru", "bilstm", "bigru"])
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    if args.experiment in ("return", "rolling"):
        test_regression(args)
    elif args.experiment == "turning_point":
        test_turning_point(args)
    elif args.experiment == "comm":
        test_comm(args)


if __name__ == "__main__":
    main()
