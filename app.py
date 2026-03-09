from flask import Flask, render_template, request, redirect, url_for, jsonify
from src.models.process import Process
from src.algorithms.fcfs import FCFSScheduler
from src.algorithms.sjf import SJFScheduler
from src.algorithms.srtf import SRTFScheduler
from src.algorithms.priority import PriorityScheduler
from src.algorithms.round_robin import RoundRobinScheduler
from src.utils.metrics import calculate_metrics
from src.utils.visualization import get_gantt_chart_base64
from src.ml.svm_predictor import predict_burst_time, predict_burst_time_batch, train

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Existing routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        algo    = request.form.get('algorithm')
        quantum = int(request.form.get('quantum', 2))

        pids       = request.form.getlist('pid[]')
        arrivals   = request.form.getlist('arrival[]')
        bursts     = request.form.getlist('burst[]')
        priorities = request.form.getlist('priority[]')

        processes = []
        for i in range(len(pids)):
            if pids[i]:
                processes.append(Process(
                    int(pids[i]),
                    int(arrivals[i]),
                    int(bursts[i]),
                    int(priorities[i])
                ))

        if not processes:
            return "No processes to schedule", 400

        scheduler = None
        if algo == "FCFS":
            scheduler = FCFSScheduler()
        elif algo == "SJF":
            scheduler = SJFScheduler()
        elif algo == "SRTF":
            scheduler = SRTFScheduler()
        elif algo == "Priority":
            scheduler = PriorityScheduler(preemptive=True)
        elif algo == "Priority-NP":
            scheduler = PriorityScheduler(preemptive=False)
        elif algo == "RR":
            scheduler = RoundRobinScheduler(time_quantum=quantum)
        else:
            return "Invalid Algorithm", 400

        for p in processes:
            scheduler.add_process(p)

        scheduler.run()

        execution_log = scheduler.get_timeline()
        metrics       = calculate_metrics(scheduler.get_processes(), execution_log)
        gantt_chart   = get_gantt_chart_base64(execution_log, title=f"Gantt Chart - {algo}")

        return render_template(
            'index.html',
            metrics=metrics,
            gantt_chart=gantt_chart,
            algorithm=algo,
            processes=processes
        )

    except Exception as e:
        return f"An error occurred: {e}", 500


# ─────────────────────────────────────────────────────────────────────────────
# SVM – Burst Time Prediction  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict-burst', methods=['POST'])
def predict_burst():
    """
    Predict the CPU burst time for one or more processes using the trained SVR.

    Accepts JSON body:
      Single process  → { "memory_percent": 1.5, "num_threads": 4, ... }
      Multiple procs  → { "processes": [ {...}, {...} ] }

    Returns:
      { "predicted_burst_time": 7.23 }
      or
      { "predicted_burst_times": [7.23, 2.11, ...] }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    try:
        # ── batch mode ──────────────────────────────────────────────────────
        if "processes" in data:
            records = data["processes"]
            if not isinstance(records, list) or len(records) == 0:
                return jsonify({"error": "'processes' must be a non-empty list"}), 400

            results = predict_burst_time_batch(records)
            return jsonify({
                "predicted_burst_times": results,
                "unit": "seconds",
                "model": "SVM (SVR, RBF kernel)"
            })

        # ── single mode ─────────────────────────────────────────────────────
        bt = predict_burst_time(
            memory_percent            = float(data.get("memory_percent",             0)),
            num_threads               = int(data.get("num_threads",                  1)),
            io_read_count             = int(data.get("io_read_count",                0)),
            io_write_count            = int(data.get("io_write_count",               0)),
            io_read_bytes             = int(data.get("io_read_bytes",                0)),
            io_write_bytes            = int(data.get("io_write_bytes",               0)),
            num_ctx_switches_voluntary= int(data.get("num_ctx_switches_voluntary",   0)),
            nice                      = int(data.get("nice",                         0)),
        )
        return jsonify({
            "predicted_burst_time": bt,
            "unit": "seconds",
            "model": "SVM (SVR, RBF kernel)"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/train-svm', methods=['POST'])
def train_svm():
    """
    Re-train the SVM model on the latest process_data.csv and return metrics.
    Useful after collecting new process data.
    """
    try:
        metrics = train()
        return jsonify({"status": "Model trained successfully", "metrics": metrics})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/svm-info', methods=['GET'])
def svm_info():
    """Return metadata about the SVM predictor (features, description)."""
    return jsonify({
        "model": "Support Vector Regression (SVR)",
        "kernel": "RBF",
        "hyperparameters": {"C": 10, "epsilon": 0.1, "gamma": "scale"},
        "target": "CPU Burst Time (seconds) = cpu_times_user + cpu_times_system",
        "features": [
            "memory_percent",
            "num_threads",
            "io_read_count",
            "io_write_count",
            "io_read_bytes",
            "io_write_bytes",
            "num_ctx_switches_voluntary",
            "nice"
        ],
        "preprocessing": "StandardScaler + log1p(target)",
        "training_data": "process_data.csv (real Windows process snapshots)"
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)