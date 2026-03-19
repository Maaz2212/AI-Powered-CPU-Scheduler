"""
Exponential Average (EMMA) Burst Time Predictor
=================================================
This is the TRADITIONAL method used by real operating systems (since the 1970s)
to predict the next CPU burst time of a process.

Formula (from Silberschatz OS textbook, Chapter 5):
    τ(n+1) = α × t(n) + (1-α) × τ(n)

Where:
    τ(n+1) = prediction for the NEXT burst
    t(n)   = ACTUAL burst time of the nth burst
    τ(n)   = prediction that was made for the nth burst
    α      = smoothing factor between 0 and 1 (typically 0.5)

Intuition:
    α = 1.0 → only trust the most recent burst (no history)
    α = 0.0 → only trust old predictions (ignore recent data)
    α = 0.5 → equal weight to recent and historical (standard)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────────
_BASE_DIR  = Path(__file__).parent
DATA_PATH  = _BASE_DIR / "process_data.csv"
STATE_PATH = _BASE_DIR / "emma_state.pkl"   # persists per-process history


class ExponentialAveragePredictor:
    """
    Predicts CPU burst time using the Exponential Moving Average formula.

    Each process is tracked by name. The predictor remembers the last
    prediction for each process name and updates it when the actual
    burst time is reported.

    This mirrors how a real OS scheduler maintains per-process burst
    history in the Process Control Block (PCB).
    """

    def __init__(self, alpha: float = 0.5, initial_estimate: float = None):
        """
        Parameters
        ----------
        alpha            : smoothing factor (0 < α ≤ 1). Default 0.5
        initial_estimate : τ_0 for unseen processes. If None, uses dataset mean.
        """
        self.alpha            = alpha
        self.initial_estimate = initial_estimate  # set during fit()
        # maps process_name → last predicted burst time (τ_n)
        self._history: dict[str, float] = {}

    # ── Training / fitting ──────────────────────────────────────────────────

    def fit(self, csv_path: Path = DATA_PATH) -> dict:
        """
        Replay historical process data to build per-process τ history.
        Also evaluates prediction accuracy on the full dataset.

        Returns evaluation metrics dict.
        """
        df = pd.read_csv(csv_path)
        df["burst_time"] = df["cpu_times_user"] + df["cpu_times_system"]
        df = df.sort_values("timestamp").reset_index(drop=True)

        if self.initial_estimate is None:
            self.initial_estimate = float(df["burst_time"].mean())

        predictions = []
        actuals     = []

        for name, group in df.groupby("name"):
            tau = self.initial_estimate          # τ_0 for this process
            for idx, row in group.iterrows():
                actual = row["burst_time"]
                predictions.append(tau)
                actuals.append(actual)
                # Update: τ(n+1) = α·t(n) + (1-α)·τ(n)
                tau = self.alpha * actual + (1 - self.alpha) * tau

            # Save final τ for future predictions
            self._history[name] = tau

        predictions = np.array(predictions)
        actuals     = np.array(actuals)

        metrics = {
            "MAE (seconds)":  round(float(mean_absolute_error(actuals, predictions)), 4),
            "RMSE (seconds)": round(float(np.sqrt(mean_squared_error(actuals, predictions))), 4),
            "R² Score":       round(float(r2_score(actuals, predictions)), 4),
            "Alpha (α)":      self.alpha,
            "Samples":        len(actuals),
        }

        # Persist history
        self._save()
        return metrics

    # ── Prediction ──────────────────────────────────────────────────────────

    def predict(self, process_name: str) -> float:
        """
        Predict the next burst time for a process.

        If the process has been seen before, returns the last τ for it.
        If unseen, returns the initial_estimate (dataset mean).
        """
        return self._history.get(process_name, self.initial_estimate or 21.86)

    def update(self, process_name: str, actual_burst: float):
        """
        Update the history after a process actually runs.
        Call this AFTER a process finishes to keep predictions current.

        τ(n+1) = α·t(n) + (1-α)·τ(n)
        """
        tau_n = self._history.get(process_name, self.initial_estimate or 21.86)
        tau_next = self.alpha * actual_burst + (1 - self.alpha) * tau_n
        self._history[process_name] = tau_next
        self._save()

    # ── Persistence ─────────────────────────────────────────────────────────

    def _save(self, path: Path = STATE_PATH):
        with open(path, "wb") as f:
            pickle.dump({
                "alpha":            self.alpha,
                "initial_estimate": self.initial_estimate,
                "history":          self._history,
            }, f)

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "ExponentialAveragePredictor":
        """Load a previously fitted predictor from disk."""
        if not path.exists():
            raise FileNotFoundError(f"No saved state at {path}. Call fit() first.")
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(alpha=data["alpha"], initial_estimate=data["initial_estimate"])
        obj._history = data["history"]
        return obj


# ── Module-level convenience functions ──────────────────────────────────────

def train_emma(csv_path: Path = DATA_PATH, alpha: float = 0.5) -> dict:
    """Train and persist the exponential average predictor. Returns metrics."""
    predictor = ExponentialAveragePredictor(alpha=alpha)
    metrics   = predictor.fit(csv_path)
    print("[EMMA] Fitted and saved.")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def predict_emma(process_name: str) -> float:
    """
    Predict burst time for a named process using the fitted EMMA model.
    Falls back to training if no saved state exists.
    """
    try:
        predictor = ExponentialAveragePredictor.load()
    except FileNotFoundError:
        print("[EMMA] No state found — training now…")
        train_emma()
        predictor = ExponentialAveragePredictor.load()
    return predictor.predict(process_name)


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exponential Average Burst Predictor")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("train", help="Fit on process_data.csv")

    p = sub.add_parser("predict", help="Predict burst for a named process")
    p.add_argument("name", type=str, help="Process name e.g. chrome.exe")

    args = parser.parse_args()
    if args.cmd == "train":
        train_emma()
    elif args.cmd == "predict":
        bt = predict_emma(args.name)
        print(f"Predicted Burst Time for '{args.name}': {bt:.4f} seconds")
    else:
        parser.print_help()