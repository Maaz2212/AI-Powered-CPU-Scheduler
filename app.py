from flask import Flask, render_template, request, jsonify
from src.models.process import Process
from src.algorithms.fcfs import FCFSScheduler
from src.algorithms.sjf import SJFScheduler
from src.algorithms.srtf import SRTFScheduler
from src.algorithms.priority import PriorityScheduler
from src.algorithms.round_robin import RoundRobinScheduler
from src.algorithms.cfs import CFSScheduler
from src.utils.metrics import calculate_metrics
from src.utils.visualization import get_gantt_chart_base64
from src.ml.rf_predictor import (
    predict_burst_time       as rf_predict,
    predict_burst_time_batch as rf_predict_batch,
    train                    as rf_train,
)
from src.ml.gb_predictor import (
    predict_burst_time       as gb_predict,
    predict_burst_time_batch as gb_predict_batch,
    train                    as gb_train,
)
from src.ml.exponential_avg import train_emma, predict_emma

app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_processes(form):
    pids       = form.getlist('pid[]')
    arrivals   = form.getlist('arrival[]')
    bursts     = form.getlist('burst[]')
    priorities = form.getlist('priority[]')
    processes  = []
    for i in range(len(pids)):
        if pids[i]:
            processes.append(Process(
                int(pids[i]),
                int(arrivals[i]),
                int(bursts[i]),
                int(priorities[i])
            ))
    return processes


def _build_scheduler(algo, quantum):
    mapping = {
        'FCFS':        FCFSScheduler(),
        'SJF':         SJFScheduler(),
        'SRTF':        SRTFScheduler(),
        'Priority':    PriorityScheduler(preemptive=True),
        'Priority-NP': PriorityScheduler(preemptive=False),
        'RR':          RoundRobinScheduler(time_quantum=quantum),
        'CFS':         CFSScheduler(),
    }
    return mapping.get(algo)


# ─────────────────────────────────────────────────────────────────────────────
# Core routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        algo      = request.form.get('algorithm')
        quantum   = int(request.form.get('quantum', 2))
        processes = _parse_processes(request.form)

        if not processes:
            return 'No processes to schedule', 400

        scheduler = _build_scheduler(algo, quantum)
        if not scheduler:
            return 'Invalid algorithm', 400

        for p in processes:
            scheduler.add_process(p)
        scheduler.run()

        execution_log = scheduler.get_timeline()
        metrics       = calculate_metrics(scheduler.get_processes(), execution_log)
        gantt_chart   = get_gantt_chart_base64(
            execution_log, title=f'Gantt Chart — {algo}'
        )

        return render_template(
            'index.html',
            metrics=metrics,
            gantt_chart=gantt_chart,
            algorithm=algo,
            processes=processes,
        )
    except Exception as e:
        return f'An error occurred: {e}', 500


# ─────────────────────────────────────────────────────────────────────────────
# Burst Time Prediction
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/predict-burst', methods=['POST'])
def predict_burst():
    """
    Predict CPU burst time using all 3 models simultaneously for comparative UI.
    Expects: { name, memory_percent, io_read_bytes, num_threads, num_ctx_switches_voluntary }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Invalid or missing JSON body'}), 400

    try:
        # Parse common inputs
        name = data.get('name', 'unknown')
        memory_percent = float(data.get('memory_percent', 0))
        num_threads = int(data.get('num_threads', 1))
        io_read_bytes = int(data.get('io_read_bytes', 0))
        num_ctx_switches_voluntary = int(data.get('num_ctx_switches_voluntary', 0))

        # 1. EMMA
        emma_bt = predict_emma(name)

        # 2. Gradient Boosting
        gb_bt = gb_predict(
            num_ctx_switches_voluntary=num_ctx_switches_voluntary,
            memory_percent=memory_percent,
            io_read_bytes=io_read_bytes,
            num_threads=num_threads
        )

        # 3. Random Forest
        rf_bt = rf_predict(
            num_ctx_switches_voluntary=num_ctx_switches_voluntary,
            memory_percent=memory_percent,
            io_read_bytes=io_read_bytes,
            num_threads=num_threads
        )

        return jsonify({
            'rf': rf_bt,
            'gb': gb_bt,
            'emma': emma_bt,
            'unit': 'seconds'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Training routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/train-rf', methods=['POST'])
def train_rf():
    try:
        metrics = rf_train()
        return jsonify({'status': 'Random Forest trained', 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-gb', methods=['POST'])
def train_gb():
    try:
        metrics = gb_train()
        return jsonify({'status': 'Gradient Boosting trained', 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-emma', methods=['POST'])
def train_emma_route():
    try:
        metrics = train_emma()
        return jsonify({'status': 'EMMA fitted', 'metrics': metrics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train-all', methods=['POST'])
def train_all():
    """Fit / train all three models in one call. Used by the UI button."""
    try:
        emma_m = train_emma()
        gb_m   = gb_train()
        rf_m   = rf_train()
        return jsonify({
            'status': 'All models trained successfully',
            'EMMA':   emma_m,
            'GB':     gb_m,
            'RF':     rf_m,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Info route
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'models': {
            'emma': {
                'name':    'Exponential Average (EMMA)',
                'formula': 'τ(n+1) = α·t(n) + (1-α)·τ(n)',
                'alpha':   0.5,
                'r2':      -0.18,
                'input':   'process name',
            },
            'gb': {
                'name':   'Gradient Boosting Regressor',
                'kernel': 'Trees',
                'r2':     'See actual metrics after training',
                'mae':    'See actual metrics after training',
                'input':  '8 process features',
            },
            'rf': {
                'name':        'Random Forest',
                'n_estimators': 100,
                'r2':           0.9999,
                'mae':          0.06,
                'input':        '8 process features',
                'top_feature':  'num_ctx_switches_voluntary (71.5%)',
            },
        }
    })


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)