from flask import Flask, render_template, request, redirect, url_for
from src.models.process import Process
from src.algorithms.fcfs import FCFSScheduler
from src.algorithms.sjf import SJFScheduler
from src.algorithms.srtf import SRTFScheduler
from src.algorithms.priority import PriorityScheduler
from src.algorithms.round_robin import RoundRobinScheduler
from src.utils.metrics import calculate_metrics
from src.utils.visualization import get_gantt_chart_base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Extract form data
        algo = request.form.get('algorithm')
        quantum = int(request.form.get('quantum', 2))
        
        # Parse processes
        # Form has n rows of pid, arrival, burst, priority
        # We need to iterate through them.
        # Assuming form fields are arrays like pid[], arrival[], etc.
        pids = request.form.getlist('pid[]')
        arrivals = request.form.getlist('arrival[]')
        bursts = request.form.getlist('burst[]')
        priorities = request.form.getlist('priority[]')
        
        processes = []
        for i in range(len(pids)):
            if pids[i]: # Ensure pid exists
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

        # Load and run
        original_processes = [p for p in processes] # Keep a copy if needed, but Process objects are mutable
        # Actually, scheduler stores references.
        for p in processes:
            scheduler.add_process(p)
            
        scheduler.run()
        
        execution_log = scheduler.get_timeline()
        metrics = calculate_metrics(scheduler.get_processes(), execution_log)
        
        # Generate Chart
        gantt_chart = get_gantt_chart_base64(execution_log, title=f"Gantt Chart - {algo}")
        
        return render_template('index.html', 
                               metrics=metrics, 
                               gantt_chart=gantt_chart, 
                               algorithm=algo,
                               processes=processes) # Pass back processes to repopulate form if needed

                               
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
