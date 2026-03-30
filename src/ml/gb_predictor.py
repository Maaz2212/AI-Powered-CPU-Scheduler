"""
Gradient Boosting CPU Burst Time Predictor
========================================
Replaces the SVM model with a Gradient Boosting Regressor for better accuracy
and faster training times. Features are identical to RF.
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent
DATA_PATH  = _BASE_DIR / "process_data.csv"
MODEL_PATH = _BASE_DIR / "gb_model.pkl"

# ── feature columns ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "num_ctx_switches_voluntary",
    "memory_percent",
    "io_read_bytes",
    "num_threads",
]
TARGET_COL = "burst_time"


def _load_and_prepare(csv_path: Path):
    df = pd.read_csv(csv_path)
    df[TARGET_COL] = df["cpu_times_user"] + df["cpu_times_system"]
    X = df[FEATURE_COLS].fillna(0)
    y = np.log1p(df[TARGET_COL])
    return X, y, df


def train(csv_path: Path = DATA_PATH, save_path: Path = MODEL_PATH) -> dict:
    """
    Train a Gradient Boosting Regressor and persist it.
    Returns evaluation metrics on both train and test sets.
    """
    X, y, _ = _load_and_prepare(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds_log_train   = model.predict(X_train)
    y_train_orig = np.expm1(y_train)
    preds_orig_train  = np.expm1(preds_log_train)

    preds_log_test   = model.predict(X_test)
    y_test_orig = np.expm1(y_test)
    preds_orig_test  = np.expm1(preds_log_test)

    importances = {
        col: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLS, model.feature_importances_)
    }

    metrics = {
        "Train MAE (s)": round(float(mean_absolute_error(y_train_orig, preds_orig_train)), 4),
        "Test MAE (s)":  round(float(mean_absolute_error(y_test_orig, preds_orig_test)), 4),
        "Train R²":      round(float(r2_score(y_train, preds_log_train)), 4),
        "Test R²":       round(float(r2_score(y_test, preds_log_test)), 4),
        "Train samples": len(X_train),
        "Test samples":  len(X_test),
        "Feature importances": importances,
    }

    bundle = {"model": model, "features": FEATURE_COLS, "importances": importances}
    with open(save_path, "wb") as f:
        pickle.dump(bundle, f)

    print("[GB] Model trained and saved to", save_path)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


def _load_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        print("[GB] No saved model — training now…")
        train()
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_burst_time(
    num_ctx_switches_voluntary: int = 0,
    memory_percent: float = 0.0,
    io_read_bytes: int = 0,
    num_threads: int = 1,
    model_path: Path = MODEL_PATH,
) -> float:
    bundle = _load_bundle(model_path)
    model  = bundle["model"]

    row = np.array([[
        num_ctx_switches_voluntary, memory_percent, io_read_bytes, num_threads
    ]], dtype=float)

    pred_log   = model.predict(row)[0]
    burst_time = float(np.expm1(pred_log))
    return max(burst_time, 0.001)


def predict_burst_time_batch(
    records: list[dict],
    model_path: Path = MODEL_PATH,
) -> list[float]:
    bundle = _load_bundle(model_path)
    model  = bundle["model"]

    rows = [[float(rec.get(col, 0)) for col in FEATURE_COLS] for rec in records]
    X    = np.array(rows, dtype=float)

    preds_log = model.predict(X)
    return [max(float(np.expm1(p)), 0.001) for p in preds_log]


def get_feature_importances(model_path: Path = MODEL_PATH) -> dict:
    bundle = _load_bundle(model_path)
    return bundle.get("importances", {})

if __name__ == "__main__":
    train()
