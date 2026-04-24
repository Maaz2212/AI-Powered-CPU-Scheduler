from flask import Flask, render_template, request, jsonify
import io
import csv
import math
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

import time

@app.route('/')
def index():
    return render_template('index.html', cache_buster=int(time.time()))


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

        # ML Models predict MICRO-BURST time (per context switch).
        # We multiply by (switches + 1) to get the predicted TOTAL CPU time for its lifetime.
        gb_bt *= (num_ctx_switches_voluntary + 1)
        rf_bt *= (num_ctx_switches_voluntary + 1)

        return jsonify({
            'rf': rf_bt,
            'gb': gb_bt,
            'emma': emma_bt,
            'unit': 'seconds'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Accept a CSV file upload and run all 3 models on each row.
    CSV columns (required): name, num_ctx_switches_voluntary, memory_percent, io_read_bytes, num_threads
    CSV column (optional): actual_burst
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send form-data with key "file".'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv'}), 400

    try:
        stream = io.StringIO(f.stream.read().decode('utf-8'))
        reader = csv.DictReader(stream)

        required = {'name', 'num_ctx_switches_voluntary', 'memory_percent', 'io_read_bytes', 'num_threads'}
        if not required.issubset(set(reader.fieldnames or [])):
            return jsonify({'error': f'CSV must contain columns: {", ".join(required)}'}), 400

        results = []
        for row in reader:
            name = row['name'].strip()
            ctx  = int(float(row['num_ctx_switches_voluntary']))
            mem  = float(row['memory_percent'])
            iob  = int(float(row['io_read_bytes']))
            thr  = int(float(row['num_threads']))
            actual = float(row['actual_burst']) if 'actual_burst' in row and row['actual_burst'].strip() else None

            rf_bt   = rf_predict(num_ctx_switches_voluntary=ctx, memory_percent=mem, io_read_bytes=iob, num_threads=thr)
            gb_bt   = gb_predict(num_ctx_switches_voluntary=ctx, memory_percent=mem, io_read_bytes=iob, num_threads=thr)
            emma_bt = predict_emma(name)

            # Convert predicted microbursts to total CPU lifetime
            rf_bt *= (ctx + 1)
            gb_bt *= (ctx + 1)

            entry = {
                'name':  name,
                'rf':    round(rf_bt,   4),
                'gb':    round(gb_bt,   4),
                'ensemble': round(math.sqrt(rf_bt * gb_bt), 4),
                'emma':  round(emma_bt, 4),
            }
            if actual is not None:
                entry['actual']    = round(actual, 4)
                entry['error_rf']  = round(abs(rf_bt   - actual), 4)
                entry['error_gb']  = round(abs(gb_bt   - actual), 4)
                entry['error_emma']= round(abs(emma_bt - actual), 4)
            results.append(entry)

        return jsonify({'results': results, 'count': len(results)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Full OS Simulation Pipeline (RR vs AI-SJF)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/simulate-batch', methods=['POST'])
def simulate_batch():
    """
    Accept a CSV file upload and run a full dual-scheduler simulation.
    Runs Round Robin (quantum=4) vs AI-SJF (RF-predicted burst times).
    Returns:
      - rr_gantt_chart, sjf_gantt_chart: base64-encoded PNG images
      - rr_metrics,  sjf_metrics:   per-process turnaround / waiting times
      - summary:     aggregate comparison (avg TAT, avg WT, throughput)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Send form-data with key "file".'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a .csv'}), 400

    try:
        stream  = io.StringIO(f.stream.read().decode('utf-8'))
        reader  = csv.DictReader(stream)
        rows    = list(reader)

        if not rows:
            return jsonify({'error': 'CSV is empty'}), 400

        required = {'name', 'num_ctx_switches_voluntary', 'memory_percent',
                    'io_read_bytes', 'num_threads'}
        if not required.issubset(set(reader.fieldnames or [])):
            return jsonify({
                'error': f'CSV must contain columns: {", ".join(required)}'
            }), 400

        # ── Build process lists ─────────────────────────────────────────────
        rr_processes  = []
        sjf_processes = []
        process_names = []

        for idx, row in enumerate(rows):
            name = row['name'].strip()
            ctx  = int(float(row.get('num_ctx_switches_voluntary', 0)))
            mem  = float(row.get('memory_percent', 0))
            iob  = int(float(row.get('io_read_bytes', 0)))
            thr  = int(float(row.get('num_threads', 1)))

            # Compute both predictions
            rf_micro = rf_predict(num_ctx_switches_voluntary=ctx, memory_percent=mem, io_read_bytes=iob, num_threads=thr)
            gb_micro = gb_predict(num_ctx_switches_voluntary=ctx, memory_percent=mem, io_read_bytes=iob, num_threads=thr)
            
            # Use Geometric Mean for Ensemble. Burst times are log-normally distributed. 
            # Arithmetic mean heavily biases toward the larger outlier. Geometric mean gracefully balances them!
            import math
            ai_micro_burst_s = math.sqrt(rf_micro * gb_micro)
            # Scale up to total burst time for realistic simulation
            ai_burst_s = ai_micro_burst_s * (ctx + 1)

            # Use actual_burst if supplied; else fall back to ai_burst for RR
            actual_s = float(row.get('actual_burst') or 0) or ai_burst_s

            # Convert to integer milliseconds for the scheduler (min 1)
            rr_burst  = max(1, round(actual_s * 1000))
            sjf_burst = max(1, round(ai_burst_s * 1000))

            # All processes arrive at t=0 to allow pure SJF sorting immediately
            arrival = 0
            process_names.append(name)

            rr_processes.append(Process(
                pid=idx + 1,
                arrival_time=arrival,
                burst_time=rr_burst,
                priority=1,
            ))
            sjf_processes.append(Process(
                pid=idx + 1,
                arrival_time=arrival,
                burst_time=sjf_burst,
                priority=1,
            ))

        # ── Limit to first 40 processes for frontend Gantt readability ──────
        cap = min(len(rr_processes), 40)
        rr_processes  = rr_processes[:cap]
        sjf_processes = sjf_processes[:cap]
        process_names = process_names[:cap]

        # ── Run Round Robin (quantum = 4 ms) ────────────────────────────────
        rr_sched = RoundRobinScheduler(time_quantum=4)
        for p in rr_processes:
            rr_sched.add_process(p)
        rr_sched.run()
        rr_log     = rr_sched.get_timeline()
        rr_metrics = calculate_metrics(rr_sched.get_processes(), rr_log)
        rr_gantt   = get_gantt_chart_base64(rr_log, 'Round Robin (q=4ms)')

        # Per-process detail for RR
        rr_detail = [
            {
                'pid':         p.pid,
                'name':        process_names[p.pid - 1],
                'burst_time':  p.burst_time,
                'tat':         p.turnaround_time,
                'wt':          p.waiting_time,
                'ct':          p.completion_time,
            }
            for p in rr_sched.get_processes()
        ]

        # ── Run SJF with AI-predicted bursts ────────────────────────────────
        sjf_sched = SJFScheduler()
        for p in sjf_processes:
            sjf_sched.add_process(p)
        sjf_sched.run()
        sjf_log     = sjf_sched.get_timeline()
        sjf_metrics = calculate_metrics(sjf_sched.get_processes(), sjf_log)
        sjf_gantt   = get_gantt_chart_base64(sjf_log, 'AI-SJF (Burst-Predicted)')

        # Per-process detail for SJF
        sjf_detail = [
            {
                'pid':         p.pid,
                'name':        process_names[p.pid - 1],
                'burst_time':  p.burst_time,
                'tat':         p.turnaround_time,
                'wt':          p.waiting_time,
                'ct':          p.completion_time,
            }
            for p in sjf_sched.get_processes()
        ]

        # Keep only 4 essential OS metrics for the UI
        def filter_metrics(m):
            allowed = ['Average Turnaround Time', 'Average Waiting Time', 'CPU Utilization', 'Context Switches']
            return {k: m.get(k, 0) for k in allowed}

        return jsonify({
            'process_names': process_names,
            'process_count': cap,
            'rr': {
                'gantt_chart': rr_gantt,
                'metrics':     filter_metrics(rr_metrics),
                'detail':      rr_detail,
            },
            'sjf': {
                'gantt_chart': sjf_gantt,
                'metrics':     filter_metrics(sjf_metrics),
                'detail':      sjf_detail,
            },
        })

    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


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
        'dataset': {
            'raw_rows':    122064,
            'unique_rows': '~65000',
            'split':       '70/30 (random_state=42)',
            'features':    4,
        },
        'models': {
            'emma': {
                'name':    'Exponential Average (EMMA)',
                'formula': 'τ(n+1) = α·t(n) + (1-α)·τ(n)',
                'alpha':   0.5,
                'r2':      0.9961,
                'input':   'process name',
            },
            'gb': {
                'name':   'Gradient Boosting Regressor',
                'kernel': 'Trees',
                'r2':     0.9686,
                'input':  '4 process features',
            },
            'rf': {
                'name':         'Random Forest',
                'n_estimators':  100,
                'r2':            0.9688,
                'input':         '4 process features',
                'top_feature':   'num_ctx_switches_voluntary (71.5%)',
            },
        }
    })


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation route
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/generate-plots', methods=['POST'])
def generate_plots_route():
    """Trigger matplotlib plot regeneration in-process."""
    try:
        from src.ml.generate_plots import generate_plots
        generate_plots()
        return jsonify({'message': '✅ All 3 plots regenerated successfully in static/plots/'})
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)