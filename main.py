"""
main.py — Experiment runner for CS515 HW4 (all parts).

Usage examples
--------------
# Part (b) — exact return forecasting
python main.py --experiment return_lstm
python main.py --experiment return_gru
python main.py --experiment return_all      # runs both

# Part (c) — rolling average return
python main.py --experiment rolling_lstm
python main.py --experiment rolling_gru
python main.py --experiment rolling_all

# Part (d) — turning point detection
python main.py --experiment tp_bilstm
python main.py --experiment tp_bigru
python main.py --experiment tp_all

# Part 2 — communication system (bonus)
python main.py --experiment comm

# Run everything
python main.py --experiment all
"""

import argparse
import torch
import numpy as np

from parameters import DataConfig, ReturnConfig, RollingConfig, TurningPointConfig, CommConfig
from models.stock_lstm  import StockLSTM
from models.stock_gru   import StockGRU
from models.comm_system import InteractiveCommSystem
from utils.dataset import (
    download_data,
    build_return_loaders,
    build_rolling_loaders,
    build_tp_loaders,
)
from utils.metrics import per_horizon_mse, mae, directional_accuracy, binary_metrics
from utils.visualization import (
    plot_loss_curves,
    plot_per_horizon_mse,
    plot_return_vs_rolling,
    plot_comm_curves,
)
from train import train_regression, evaluate_regression, train_classification, train_comm


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_preds(model, loader, device):
    import torch
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu())
            targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def _print_regression_results(tag, preds, targets, horizons):
    h_mse = per_horizon_mse(preds, targets)
    print(f"\n  ── {tag} ──")
    print(f"  Overall MSE : {h_mse.mean():.6f}  |  MAE : {mae(preds, targets):.6f}"
          f"  |  Dir.Acc : {directional_accuracy(preds, targets):.4f}")
    for d, m in zip(horizons, h_mse):
        print(f"    d={d}: {m:.6f}")
    return h_mse


# ─────────────────────────────────────────────────────────────────────────────
# Part (b) — exact return ratio
# ─────────────────────────────────────────────────────────────────────────────

def run_return(model_type: str, all_data, device):
    print(f"\n{'='*60}")
    print(f"  PART (b) — Return Forecasting | {model_type.upper()}")
    print(f"{'='*60}")

    data_cfg = DataConfig()
    ret_cfg  = ReturnConfig()
    n_feat   = len(data_cfg.features)
    n_hor    = len(data_cfg.horizons)

    train_loader, val_loader, test_loader, _ = build_return_loaders(
        all_data, data_cfg, ret_cfg.batch_size
    )

    if model_type == "lstm":
        model = StockLSTM(n_feat, ret_cfg.hidden_size, ret_cfg.num_layers, n_hor, ret_cfg.dropout)
    else:
        model = StockGRU(n_feat, ret_cfg.hidden_size, ret_cfg.num_layers, n_hor, ret_cfg.dropout)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tr_losses, val_losses = train_regression(
        model, train_loader, val_loader,
        epochs=ret_cfg.epochs, lr=ret_cfg.lr,
        weight_decay=ret_cfg.weight_decay,
        patience=ret_cfg.patience,
        device=device,
        ckpt_name=f"return_{model_type}",
    )
    plot_loss_curves(tr_losses, val_losses,
                     f"Part (b) {model_type.upper()} — MSE Loss",
                     f"return_{model_type}_loss.png")

    preds, targets = _collect_preds(model, test_loader, device)
    h_mse = _print_regression_results(f"Part (b) TEST — {model_type.upper()}", preds, targets, data_cfg.horizons)
    return h_mse, model


# ─────────────────────────────────────────────────────────────────────────────
# Part (c) — rolling average return
# ─────────────────────────────────────────────────────────────────────────────

def run_rolling(model_type: str, all_data, device):
    print(f"\n{'='*60}")
    print(f"  PART (c) — Rolling Return Forecasting | {model_type.upper()}")
    print(f"{'='*60}")

    data_cfg = DataConfig()
    ret_cfg  = ReturnConfig()
    roll_cfg = RollingConfig()
    n_feat   = len(data_cfg.features)
    n_hor    = len(data_cfg.horizons)

    train_loader, val_loader, test_loader = build_rolling_loaders(
        all_data, data_cfg, roll_cfg, roll_cfg.batch_size
    )

    if model_type == "lstm":
        model = StockLSTM(n_feat, ret_cfg.hidden_size, ret_cfg.num_layers, n_hor, ret_cfg.dropout)
    else:
        model = StockGRU(n_feat, ret_cfg.hidden_size, ret_cfg.num_layers, n_hor, ret_cfg.dropout)

    tr_losses, val_losses = train_regression(
        model, train_loader, val_loader,
        epochs=roll_cfg.epochs, lr=roll_cfg.lr,
        weight_decay=roll_cfg.weight_decay,
        patience=roll_cfg.patience,
        device=device,
        ckpt_name=f"rolling_{model_type}",
    )
    plot_loss_curves(tr_losses, val_losses,
                     f"Part (c) {model_type.upper()} Rolling — MSE Loss",
                     f"rolling_{model_type}_loss.png")

    preds, targets = _collect_preds(model, test_loader, device)
    h_mse = _print_regression_results(f"Part (c) TEST — {model_type.upper()}", preds, targets, data_cfg.horizons)
    return h_mse, model


# ─────────────────────────────────────────────────────────────────────────────
# Part (d) — turning point detection
# ─────────────────────────────────────────────────────────────────────────────

def run_turning_point(model_type: str, all_data, device):
    print(f"\n{'='*60}")
    print(f"  PART (d) — Turning Point Detection | Bi-{model_type.upper()}")
    print(f"{'='*60}")

    data_cfg = DataConfig()
    tp_cfg   = TurningPointConfig()
    n_feat   = len(data_cfg.features)

    train_loader, val_loader, test_loader, pos_weight = build_tp_loaders(
        all_data, data_cfg, tp_cfg, tp_cfg.batch_size
    )
    print(f"  Positive-class weight: {pos_weight.item():.3f}")

    if model_type == "lstm":
        model = StockLSTM(n_feat, tp_cfg.hidden_size, tp_cfg.num_layers,
                          output_size=1, dropout=tp_cfg.dropout, bidirectional=True)
    else:
        model = StockGRU(n_feat, tp_cfg.hidden_size, tp_cfg.num_layers,
                         output_size=1, dropout=tp_cfg.dropout, bidirectional=True)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    tr_losses, val_losses = train_classification(
        model, train_loader, val_loader,
        epochs=tp_cfg.epochs, lr=tp_cfg.lr,
        weight_decay=tp_cfg.weight_decay,
        patience=tp_cfg.patience,
        device=device,
        ckpt_name=f"tp_bi{model_type}",
        pos_weight=pos_weight,
    )
    plot_loss_curves(tr_losses, val_losses,
                     f"Part (d) Bi-{model_type.upper()} — BCE Loss",
                     f"tp_bi{model_type}_loss.png")

    preds, labels = _collect_preds(model, test_loader, device)
    preds = preds.squeeze(-1)
    metrics = binary_metrics(preds, labels)
    print(f"\n  ── Part (d) TEST — Bi-{model_type.upper()} ──")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — interactive communication system
# ─────────────────────────────────────────────────────────────────────────────

def run_comm(device):
    print(f"\n{'='*60}")
    print(f"  PART 2 — Interactive AWGN Communication System")
    print(f"{'='*60}")

    cfg   = CommConfig()
    model = InteractiveCommSystem(cfg)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    tr_losses, val_sers, val_mers = train_comm(
        model, epochs=cfg.epochs, lr=cfg.lr,
        batch_size=cfg.batch_size, patience=cfg.patience,
        device=device,
        alphabet_size=cfg.alphabet_size,
        seq_len=cfg.seq_len,
    )
    plot_comm_curves(tr_losses, val_sers, val_mers)

    # Final eval
    model.eval()
    n_test = 10000
    msgs = torch.randint(0, cfg.alphabet_size, (n_test, cfg.seq_len), device=device)
    with torch.no_grad():
        logits = model(msgs)
    from utils.metrics import symbol_error_rate, message_error_rate
    ser = symbol_error_rate(logits, msgs)
    mer = message_error_rate(logits, msgs)
    print(f"\n  Final test @ σ²={cfg.noise_var} | SER={ser:.4f} | MER={mer:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    "return_lstm", "return_gru", "return_all",
    "rolling_lstm", "rolling_gru", "rolling_all",
    "tp_bilstm", "tp_bigru", "tp_all",
    "comm",
    "all",
}


def main():
    parser = argparse.ArgumentParser(description="CS515 HW4 — Main runner")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS))
    args = parser.parse_args()

    device = get_device()

    exp = args.experiment
    needs_stock = exp != "comm"
    all_data = None

    if needs_stock:
        print("\nDownloading stock data …")
        data_cfg = DataConfig()
        all_data = download_data(data_cfg)

    # ── Part (b) ──
    mse_ret_lstm = mse_ret_gru = None
    if exp in ("return_lstm", "return_all", "all"):
        mse_ret_lstm, _ = run_return("lstm", all_data, device)
    if exp in ("return_gru", "return_all", "all"):
        mse_ret_gru, _  = run_return("gru",  all_data, device)

    # ── Part (c) ──
    mse_rol_lstm = mse_rol_gru = None
    if exp in ("rolling_lstm", "rolling_all", "all"):
        mse_rol_lstm, _ = run_rolling("lstm", all_data, device)
    if exp in ("rolling_gru", "rolling_all", "all"):
        mse_rol_gru, _  = run_rolling("gru",  all_data, device)

    # Combined comparison plot
    if mse_ret_lstm is not None and mse_rol_lstm is not None:
        if mse_ret_gru is None:
            mse_ret_gru = mse_ret_lstm
        if mse_rol_gru is None:
            mse_rol_gru = mse_rol_lstm
        plot_return_vs_rolling(mse_ret_lstm, mse_rol_lstm, mse_ret_gru, mse_rol_gru)
        plot_per_horizon_mse(mse_ret_lstm, mse_ret_gru,
                             "Part (b) Per-Horizon MSE — LSTM vs GRU",
                             "return_horizon_mse.png")
        plot_per_horizon_mse(mse_rol_lstm, mse_rol_gru,
                             "Part (c) Per-Horizon MSE — LSTM vs GRU",
                             "rolling_horizon_mse.png")

    # ── Part (d) ──
    if exp in ("tp_bilstm", "tp_all", "all"):
        run_turning_point("lstm", all_data, device)
    if exp in ("tp_bigru", "tp_all", "all"):
        run_turning_point("gru", all_data, device)

    # ── Part 2 ──
    if exp in ("comm", "all"):
        run_comm(device)

    print("\n✓ All requested experiments complete.")


if __name__ == "__main__":
    main()
