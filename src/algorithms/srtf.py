from src.algorithms.scheduler_base import Scheduler
from typing import Optional

class SRTFScheduler(Scheduler):
    def run(self):
        # Sort by arrival time initially
        self.processes.sort(key=lambda p: p.arrival_time)
        
        n = len(self.processes)
        current_time = 0
        completed_count = 0
        
        # To keep track of processes that have arrived and are not completed
        ready_queue = []
        
        # Index to track which processes have been added to ready queue
        process_idx = 0
        
        while completed_count < n:
            # Add newly arrived processes to ready queue
            while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
                ready_queue.append(self.processes[process_idx])
                process_idx += 1
            
            if not ready_queue:
                # No process ready, jump to next arrival
                if process_idx < n:
                    current_time = self.processes[process_idx].arrival_time
                    continue
                else:
                    # Should not happen if completed_count < n
                    break
            
            # Select process with shortest remaining time
            # Tie-breaker: Arrival time (implicit as we append to ready_queue in arrival order)
            # If we need strict FCFS on tie, min creates stability if keyed correctly.
            # But plain min on list of objects needs comparison, let's use key.
            # We want min remaining time. If tie, the one that arrived earlier (or started earlier)
            # Python's min is stable, so it picks the first one in the list which matches arrival order.
            current_process = min(ready_queue, key=lambda p: p.remaining_time)
            
            # Execute for 1 unit of time
            if current_process.first_run_time == -1:
                current_process.first_run_time = current_time
            
            if current_process.start_time == -1:
                current_process.start_time = current_time
            
            self.log_execution(current_time, current_process.pid)
            
            current_process.remaining_time -= 1
            current_time += 1
            
            if current_process.is_completed():
                current_process.update_metrics(current_time)
                ready_queue.remove(current_process)
                completed_count += 1
