# CS515 HW4 — Sequence Modeling

**Sabancı University | CS515 Deep Learning**

Sequence modelling experiments using LSTM, GRU, and Transformer networks for financial forecasting and interactive communication systems.

---

## Project Structure

```
CS515-HW4/
├── main.py              # Experiment runner (all parts)
├── train.py             # Training loops with early stopping
├── test.py              # Load checkpoint → evaluate on test set
├── parameters.py        # Dataclasses for all hyperparameters
├── requirements.txt
├── models/
│   ├── stock_lstm.py    # StockLSTM (Parts b, c, d)
│   ├── stock_gru.py     # StockGRU  (Parts b, c, d)
│   └── comm_system.py   # Transformer TX/RX (Part 2 bonus)
├── utils/
│   ├── dataset.py       # yfinance download, sliding-window datasets
│   ├── metrics.py       # MSE, MAE, Dir.Acc, SER, MER, F1 …
│   └── visualization.py # Training curves, bar charts, confusion matrices
├── results/
│   ├── figures/         # Saved plots
│   └── checkpoints/     # Best model weights
└── cs515_HW4_ColabGpu.ipynb   # Full Colab notebook (GPU)
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Experiments

### Part (a) — Dataset

Downloads AAPL, MSFT, GOOGL daily OHLC data (2020–2025) via `yfinance`.  
Chronological split: train Jan 2020–Jul 2024 | val Aug–Dec 2024 | test Jan–Dec 2025.

---

### Part (b) — Exact d-day Return Forecasting

```bash
# Train StockLSTM
python main.py --experiment return_lstm

# Train StockGRU
python main.py --experiment return_gru

# Train both
python main.py --experiment return_all
```

**Evaluate saved checkpoint:**
```bash
python test.py --experiment return --model lstm \
    --checkpoint results/checkpoints/return_lstm_best.pth
```

---

### Part (c) — Rolling-Average Return Forecasting (l=3)

```bash
python main.py --experiment rolling_lstm
python main.py --experiment rolling_gru
python main.py --experiment rolling_all
```

---

### Part (d) — Turning-Point Detection (Bi-directional LSTM/GRU)

Buy signal if any d-day return exceeds threshold γ using the **High** price.

```bash
python main.py --experiment tp_bilstm
python main.py --experiment tp_bigru
python main.py --experiment tp_all
```

**Evaluate:**
```bash
python test.py --experiment turning_point --model bilstm \
    --checkpoint results/checkpoints/tp_bilstm_best.pth
```

---

### Part 2 (Bonus) — Interactive AWGN Communication System

Transformer-based TX encoder and RX decoder trained end-to-end with T=4 rounds, σ²=0.25.

```bash
python main.py --experiment comm
```

**Evaluate (includes SNR sweep):**
```bash
python test.py --experiment comm \
    --checkpoint results/checkpoints/comm_system_best.pth
```

---

### Run All Experiments

```bash
python main.py --experiment all
```

---

## Hyperparameters

All hyperparameters are centralised in `parameters.py` as Python dataclasses.

| Config | Key params |
|---|---|
| `DataConfig` | tickers, lookback T=20, horizons d=1..5 |
| `ReturnConfig` | hidden=128, layers=2, dropout=0.2, lr=1e-3 |
| `RollingConfig` | roll_window l=3, weights [0.5, 0.3, 0.2] |
| `TurningPointConfig` | γ=0.01, bidirectional=True |
| `CommConfig` | T=4 rounds, σ²=0.25, d_model=64 |

---

## References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory," *Neural Computation* 1997.
2. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder," *EMNLP* 2014.
3. Vaswani et al., "Attention Is All You Need," *NeurIPS* 2017.
4. Kim & Mohaisen, "Deep Learning-Based Interactive Communication Systems," 2023.
5. yfinance: https://pypi.org/project/yfinance/
