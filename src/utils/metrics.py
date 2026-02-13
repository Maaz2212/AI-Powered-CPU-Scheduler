from typing import List, Dict
from src.models.process import Process

def calculate_metrics(processes: List[Process], execution_log: List[Dict]) -> Dict[str, float]:
    n = len(processes)
    if n == 0:
        return {}
    
    # Calculate total time from completion times
    max_completion_time = max(p.completion_time for p in processes) if processes else 0
    min_arrival_time = min(p.arrival_time for p in processes) if processes else 0
    total_simulation_time = max_completion_time - min_arrival_time
    if total_simulation_time == 0:
         total_simulation_time = 1 # Avoid division by zero

    total_waiting_time = sum(p.waiting_time for p in processes)
    total_turnaround_time = sum(p.turnaround_time for p in processes)
    total_response_time = sum(p.response_time for p in processes)
    
    # Calculate CPU Utilization
    # Count busy time slots from log (unique time slots)
    busy_time = len(set(entry['time'] for entry in execution_log))
    cpu_utilization = (busy_time / total_simulation_time) * 100 if total_simulation_time > 0 else 0

    avg_waiting_time = total_waiting_time / n
    avg_turnaround_time = total_turnaround_time / n
    avg_response_time = total_response_time / n

    # Fairness Index (Jain's)
    # For now, let's use waiting time for fairness calculation
    # sum(xi)^2 / (n * sum(xi^2))
    sum_waiting = total_waiting_time
    sum_sq_waiting = sum(p.waiting_time ** 2 for p in processes)
    
    fairness_index = 1.0
    if sum_sq_waiting > 0:
        fairness_index = (sum_waiting ** 2) / (n * sum_sq_waiting)

    first_arrival = min((p.arrival_time for p in processes), default=0)
    throughput = n / total_simulation_time if total_simulation_time > 0 else 0

    # Context Switches
    context_switches = 0
    if execution_log:
        sorted_log = sorted(execution_log, key=lambda x: x['time'])
        current_pid = sorted_log[0]['pid']
        for i in range(1, len(sorted_log)):
            # If time is continuous and pid changes, it's a switch
            # What if there is a gap (idle)?
            # Switch happens when the process executing changes.
            # Including Idle -> Process or Process -> Idle?
            # Usually Process A -> Process B is a switch.
            # Process A -> Process A (continuous) is not.
            # Process A -> Idle -> Process B. This involves saving A and loading B.
            # So effectively A->B transition (with or without idle) count as switch?
            # Let's count every time the PID changes in the sorted log.
            # But we must ensure the log is sorted by time.
            next_pid = sorted_log[i]['pid']
            if next_pid != current_pid:
                context_switches += 1
                current_pid = next_pid
                
    # Variance in Waiting Time
    variance_waiting_time = sum((p.waiting_time - avg_waiting_time) ** 2 for p in processes) / n

    max_waiting_time = max(p.waiting_time for p in processes)

    return {
        "Average Waiting Time": avg_waiting_time,
        "Average Turnaround Time": avg_turnaround_time,
        "Average Response Time": avg_response_time,
        "Throughput": throughput,
        "CPU Utilization": cpu_utilization,
        "Context Switches": context_switches,
        "Fairness Index": fairness_index,
        "Variance Waiting Time": variance_waiting_time,
        "Max Waiting Time": max_waiting_time
    }
