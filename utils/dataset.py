"""
utils/dataset.py — Data download, feature engineering, and PyTorch Datasets
for CS515 HW4 (Part 1).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional, Dict
import yfinance as yf

from parameters import DataConfig, ReturnConfig, RollingConfig, TurningPointConfig


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Raw data download & normalisation
# ─────────────────────────────────────────────────────────────────────────────

def download_data(cfg: DataConfig) -> Dict[str, pd.DataFrame]:
    """Download OHLC data for all tickers and return a dict {ticker: df}."""
    data = {}
    for ticker in cfg.tickers:
        df = yf.download(
            ticker,
            start=cfg.start_date,
            end=cfg.end_date,
            progress=False,
            auto_adjust=True,
        )
        # Flatten multi-index columns if yfinance returns them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Keep only OHLC
        df = df[["Open", "High", "Low", "Close"]].dropna()
        data[ticker] = df
        print(f"  {ticker}: {len(df)} trading days downloaded.")
    return data


def split_df(df: pd.DataFrame, cfg: DataConfig):
    """Chronological train / val / test split."""
    train = df[df.index <= cfg.train_end]
    val   = df[(df.index > cfg.train_end) & (df.index <= cfg.val_end)]
    test  = df[df.index > cfg.val_end]
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Part (b) — exact d-day return ratio
# ─────────────────────────────────────────────────────────────────────────────

class StockReturnDataset(Dataset):
    """
    Sliding-window dataset for d-day return ratio forecasting.

    X[i] : (T, F)   — normalised OHLC window
    y[i] : (D,)     — return ratios for d = 1 … D
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        scaler: Optional[StandardScaler],
        cfg: DataConfig,
        fit_scaler: bool = False,
    ):
        self.cfg = cfg
        T = cfg.lookback
        D = len(cfg.horizons)
        d_max = max(cfg.horizons)

        raw = price_df[cfg.features].values.astype(np.float32)   # (N_days, F)

        # Fit or transform
        if fit_scaler:
            scaler.fit(raw)
        self.scaler = scaler
        norm = scaler.transform(raw)                              # (N_days, F)

        close_raw = price_df["Close"].values.astype(np.float64).flatten()  # (N_days,)

        X_list, y_list = [], []
        for t in range(T, len(raw) - d_max):
            window = norm[t - T: t]                               # (T, F)
            p_t = float(close_raw[t - 1])                         # close at time t
            returns = np.array(
                [(float(close_raw[t - 1 + d]) - p_t) / p_t for d in cfg.horizons],
                dtype=np.float32,
            )
            X_list.append(window)
            y_list.append(returns)

        self.X = torch.from_numpy(np.stack(X_list))   # (N, T, F)
        self.y = torch.from_numpy(np.stack(y_list))   # (N, D)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Part (c) — rolling-average return ratio
# ─────────────────────────────────────────────────────────────────────────────

class StockRollingDataset(Dataset):
    """
    Like StockReturnDataset but the target is a weighted rolling-average return.

    ŷ_i^{t+d} = (Σ_{j=0}^{l-1} w_j * p_{t+d-j} - p_t) / p_t
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        scaler: Optional[StandardScaler],
        cfg: DataConfig,
        roll_cfg: RollingConfig,
        fit_scaler: bool = False,
    ):
        T = cfg.lookback
        D = len(cfg.horizons)
        d_max = max(cfg.horizons)
        l = roll_cfg.roll_window
        w = np.array(roll_cfg.weights, dtype=np.float64)
        w = w / w.sum()                                           # normalise

        raw = price_df[cfg.features].values.astype(np.float32)
        if fit_scaler:
            scaler.fit(raw)
        self.scaler = scaler
        norm = scaler.transform(raw)

        close_raw = price_df["Close"].values.astype(np.float64).flatten()

        X_list, y_list = [], []
        # Need at least l extra days after d_max to compute rolling average
        for t in range(T, len(raw) - d_max - l + 1):
            window = norm[t - T: t]
            p_t = float(close_raw[t - 1])
            returns = []
            for d in cfg.horizons:
                # prices p_{t+d}, p_{t+d-1}, ..., p_{t+d-l+1}
                p_window = close_raw[t - 1 + d: t - 1 + d + l]  # length l
                if len(p_window) < l:
                    p_window = np.pad(p_window, (0, l - len(p_window)), mode="edge")
                roll_ret = (np.dot(w, p_window) - p_t) / p_t
                returns.append(roll_ret)
            X_list.append(window)
            y_list.append(np.array(returns, dtype=np.float32))

        self.X = torch.from_numpy(np.stack(X_list))
        self.y = torch.from_numpy(np.stack(y_list))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Part (d) — turning-point (buy / pass) detection
# ─────────────────────────────────────────────────────────────────────────────

class TurningPointDataset(Dataset):
    """
    Binary classification dataset.

    Buy signal = 1  if  ∃d ∈ {1..5}: (p_max_{t+d} - p_t) / p_t > γ
    Otherwise pass = 0.

    p_max_{t+d} = High price on day t+d (maximum intraday price).
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        scaler: Optional[StandardScaler],
        cfg: DataConfig,
        tp_cfg: TurningPointConfig,
        fit_scaler: bool = False,
    ):
        T = cfg.lookback
        d_max = max(cfg.horizons)
        gamma = tp_cfg.gamma

        raw = price_df[cfg.features].values.astype(np.float32)
        if fit_scaler:
            scaler.fit(raw)
        self.scaler = scaler
        norm = scaler.transform(raw)

        close_raw = price_df["Close"].values.astype(np.float64).flatten()
        high_raw  = price_df["High"].values.astype(np.float64).flatten()

        X_list, y_list = [], []
        for t in range(T, len(raw) - d_max):
            window = norm[t - T: t]
            p_t = float(close_raw[t - 1])
            buy = 0
            for d in cfg.horizons:
                p_max = float(high_raw[t - 1 + d])
                if (p_max - p_t) / p_t > gamma:
                    buy = 1
                    break
            X_list.append(window)
            y_list.append(buy)

        self.X = torch.from_numpy(np.stack(X_list))
        self.y = torch.from_numpy(np.array(y_list, dtype=np.float32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Multi-stock dataset factory
# ─────────────────────────────────────────────────────────────────────────────

def _concat_datasets(datasets):
    """Concatenate a list of PyTorch Datasets by stacking X and y."""
    X = torch.cat([d.X for d in datasets], dim=0)
    y = torch.cat([d.y for d in datasets], dim=0)

    class _CatDS(Dataset):
        def __len__(self):
            return len(X)
        def __getitem__(self, i):
            return X[i], y[i]

    return _CatDS()


def build_return_loaders(
    all_data: Dict[str, pd.DataFrame],
    cfg: DataConfig,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[StandardScaler]]:
    """Build train / val / test DataLoaders for Part (b)."""
    train_ds_list, val_ds_list, test_ds_list = [], [], []
    scalers = []

    for ticker, df in all_data.items():
        tr, va, te = split_df(df, cfg)
        sc = StandardScaler()
        tr_ds = StockReturnDataset(tr, sc, cfg, fit_scaler=True)
        va_ds = StockReturnDataset(va, sc, cfg)
        te_ds = StockReturnDataset(te, sc, cfg)
        train_ds_list.append(tr_ds)
        val_ds_list.append(va_ds)
        test_ds_list.append(te_ds)
        scalers.append(sc)

    train_loader = DataLoader(_concat_datasets(train_ds_list), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(_concat_datasets(val_ds_list),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(_concat_datasets(test_ds_list),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, scalers


def build_rolling_loaders(
    all_data: Dict[str, pd.DataFrame],
    cfg: DataConfig,
    roll_cfg: RollingConfig,
    batch_size: int = 64,
):
    """Build train / val / test DataLoaders for Part (c)."""
    train_ds_list, val_ds_list, test_ds_list = [], [], []

    for ticker, df in all_data.items():
        tr, va, te = split_df(df, cfg)
        sc = StandardScaler()
        tr_ds = StockRollingDataset(tr, sc, cfg, roll_cfg, fit_scaler=True)
        va_ds = StockRollingDataset(va, sc, cfg, roll_cfg)
        te_ds = StockRollingDataset(te, sc, cfg, roll_cfg)
        train_ds_list.append(tr_ds)
        val_ds_list.append(va_ds)
        test_ds_list.append(te_ds)

    train_loader = DataLoader(_concat_datasets(train_ds_list), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(_concat_datasets(val_ds_list),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(_concat_datasets(test_ds_list),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def build_tp_loaders(
    all_data: Dict[str, pd.DataFrame],
    cfg: DataConfig,
    tp_cfg: TurningPointConfig,
    batch_size: int = 64,
):
    """Build train / val / test DataLoaders for Part (d)."""
    train_ds_list, val_ds_list, test_ds_list = [], [], []

    for ticker, df in all_data.items():
        tr, va, te = split_df(df, cfg)
        sc = StandardScaler()
        tr_ds = TurningPointDataset(tr, sc, cfg, tp_cfg, fit_scaler=True)
        va_ds = TurningPointDataset(va, sc, cfg, tp_cfg)
        te_ds = TurningPointDataset(te, sc, cfg, tp_cfg)
        train_ds_list.append(tr_ds)
        val_ds_list.append(va_ds)
        test_ds_list.append(te_ds)

    # Compute positive class weight for BCEWithLogitsLoss
    all_labels = torch.cat([d.y for d in train_ds_list])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)])

    train_loader = DataLoader(_concat_datasets(train_ds_list), batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(_concat_datasets(val_ds_list),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(_concat_datasets(test_ds_list),  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, pos_weight
