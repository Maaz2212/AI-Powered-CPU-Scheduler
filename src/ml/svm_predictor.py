"""
SVM-based CPU Burst Time Predictor
====================================
Uses a Support Vector Regression (SVR) model trained on real process data
(process_data.csv) to predict the CPU burst time of a new process.

Features used for prediction:
  - memory_percent          : % of RAM used by the process
  - num_threads             : number of threads
  - io_read_count           : total I/O read operations
  - io_write_count          : total I/O write operations
  - io_read_bytes           : total bytes read from disk
  - io_write_bytes          : total bytes written to disk
  - num_ctx_switches_voluntary : voluntary context switches
  - nice                    : process nice / priority value

Target: burst_time = cpu_times_user + cpu_times_system  (in seconds)
        A log1p transform is applied before training to handle the heavy
        right-skewed distribution; predictions are back-transformed via expm1.
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
_BASE_DIR = Path(__file__).parent
DATA_PATH = _BASE_DIR / "process_data.csv"
MODEL_PATH = _BASE_DIR / "svm_model.pkl"

# ── constants ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "memory_percent",
    "num_threads",
    "io_read_count",
    "io_write_count",
    "io_read_bytes",
    "io_write_bytes",
    "num_ctx_switches_voluntary",
    "nice",
]
TARGET_COL = "burst_time"


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV, engineer the target column, return (X, y)."""
    df = pd.read_csv(csv_path)

    # Derive burst time from accumulated CPU user + system times
    df[TARGET_COL] = df["cpu_times_user"] + df["cpu_times_system"]

    X = df[FEATURE_COLS].fillna(0)
    y = np.log1p(df[TARGET_COL])          # log-transform to reduce skew
    return X, y


def train(csv_path: Path = DATA_PATH, save_path: Path = MODEL_PATH) -> dict:
    """
    Train an SVR model on *csv_path* and persist it to *save_path*.

    Returns a dict with evaluation metrics on the held-out test set.
    """
    X, y = _load_and_prepare(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # RBF-kernel SVR — C and epsilon tuned empirically on this dataset
    model = SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")
    model.fit(X_train_s, y_train)

    # ── evaluate ────────────────────────────────────────────────────────────────
    preds_log   = model.predict(X_test_s)
    y_test_orig = np.expm1(y_test)
    preds_orig  = np.expm1(preds_log)

    metrics = {
        "MAE (seconds)":  round(float(mean_absolute_error(y_test_orig, preds_orig)), 4),
        "RMSE (seconds)": round(float(np.sqrt(mean_squared_error(y_test_orig, preds_orig))), 4),
        "R² Score":       round(float(r2_score(y_test, preds_log)), 4),
        "Train samples":  len(X_train),
        "Test samples":   len(X_test),
    }

    # ── persist ─────────────────────────────────────────────────────────────────
    bundle = {"model": model, "scaler": scaler, "features": FEATURE_COLS}
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)

    print("[SVM] Model trained and saved to", save_path)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Inference
# ══════════════════════════════════════════════════════════════════════════════

def _load_bundle(model_path: Path = MODEL_PATH) -> dict:
    """Load the persisted model bundle, training first if it does not exist."""
    if not model_path.exists():
        print("[SVM] No saved model found — training now …")
        train()
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_burst_time(
    memory_percent: float = 0.0,
    num_threads: int = 1,
    io_read_count: int = 0,
    io_write_count: int = 0,
    io_read_bytes: int = 0,
    io_write_bytes: int = 0,
    num_ctx_switches_voluntary: int = 0,
    nice: int = 0,
    model_path: Path = MODEL_PATH,
) -> float:
    """
    Predict the CPU burst time (seconds) for a single process.

    Parameters mirror the feature columns used during training.
    Returns the predicted burst time as a positive float.
    """
    bundle = _load_bundle(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    row = np.array([[
        memory_percent,
        num_threads,
        io_read_count,
        io_write_count,
        io_read_bytes,
        io_write_bytes,
        num_ctx_switches_voluntary,
        nice,
    ]], dtype=float)

    row_scaled = scaler.transform(row)
    pred_log   = model.predict(row_scaled)[0]
    burst_time = float(np.expm1(pred_log))

    # Clamp to a sensible minimum (at least 0.001 s)
    return max(burst_time, 0.001)


def predict_burst_time_batch(
    records: list[dict],
    model_path: Path = MODEL_PATH,
) -> list[float]:
    """
    Predict burst times for a list of process-feature dicts.

    Each dict should contain keys matching FEATURE_COLS (missing keys → 0).

    Example
    -------
    >>> results = predict_burst_time_batch([
    ...     {"memory_percent": 1.5, "num_threads": 8},
    ...     {"memory_percent": 0.2, "num_threads": 2, "io_read_bytes": 500000},
    ... ])
    """
    bundle = _load_bundle(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    rows = []
    for rec in records:
        rows.append([float(rec.get(col, 0)) for col in FEATURE_COLS])

    X = np.array(rows, dtype=float)
    X_scaled = scaler.transform(X)
    preds_log = model.predict(X_scaled)
    return [max(float(np.expm1(p)), 0.001) for p in preds_log]


# ══════════════════════════════════════════════════════════════════════════════
# CLI helper
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SVM Burst Time Predictor — train or predict"
    )
    sub = parser.add_subparsers(dest="cmd")

    # -- train sub-command
    sub.add_parser("train", help="Train and save the SVR model")

    # -- predict sub-command
    p = sub.add_parser("predict", help="Predict burst time for one process")
    p.add_argument("--memory",     type=float, default=0.0,  help="memory_percent")
    p.add_argument("--threads",    type=int,   default=1,    help="num_threads")
    p.add_argument("--io-reads",   type=int,   default=0,    help="io_read_count")
    p.add_argument("--io-writes",  type=int,   default=0,    help="io_write_count")
    p.add_argument("--read-bytes", type=int,   default=0,    help="io_read_bytes")
    p.add_argument("--write-bytes",type=int,   default=0,    help="io_write_bytes")
    p.add_argument("--ctx-sw",     type=int,   default=0,    help="num_ctx_switches_voluntary")
    p.add_argument("--nice",       type=int,   default=0,    help="nice value")

    args = parser.parse_args()

    if args.cmd == "train":
        train()
    elif args.cmd == "predict":
        bt = predict_burst_time(
            memory_percent=args.memory,
            num_threads=args.threads,
            io_read_count=args.io_reads,
            io_write_count=args.io_writes,
            io_read_bytes=args.read_bytes,
            io_write_bytes=args.write_bytes,
            num_ctx_switches_voluntary=args.ctx_sw,
            nice=args.nice,
        )
        print(f"Predicted Burst Time: {bt:.4f} seconds")
    else:
        parser.print_help()