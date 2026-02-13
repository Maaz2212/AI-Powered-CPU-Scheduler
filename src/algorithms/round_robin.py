from src.algorithms.scheduler_base import Scheduler
from collections import deque

class RoundRobinScheduler(Scheduler):
    def __init__(self, time_quantum: int = 2):
        super().__init__()
        self.time_quantum = time_quantum

    def run(self):
        self.processes.sort(key=lambda p: p.arrival_time)
        n = len(self.processes)
        current_time = 0
        completed_count = 0
        
        ready_queue = deque()
        process_idx = 0
        
        # Processes that are currently in the queue
        in_queue = [False] * n 
        
        # Add processes arriving at time 0
        while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
             ready_queue.append(self.processes[process_idx])
             in_queue[process_idx] = True
             process_idx += 1
        
        while completed_count < n:
            if not ready_queue:
                # No process ready, jump to next arrival
                if process_idx < n:
                    current_time = self.processes[process_idx].arrival_time
                    while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
                        ready_queue.append(self.processes[process_idx])
                        in_queue[process_idx] = True
                        process_idx += 1
                    continue
                else:
                    break

            current_process = ready_queue.popleft()
            
            if current_process.first_run_time == -1:
                current_process.first_run_time = current_time
            if current_process.start_time == -1:
                current_process.start_time = current_time
                
            # Execute for time quantum or until completion
            run_time = min(self.time_quantum, current_process.remaining_time)
            
            for _ in range(run_time):
                self.log_execution(current_time, current_process.pid)
                current_time += 1
                
                # Check for new arrivals during execution
                while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
                    if not in_queue[process_idx]:
                        ready_queue.append(self.processes[process_idx])
                        in_queue[process_idx] = True
                    process_idx += 1
            
            current_process.remaining_time -= run_time
            
            if current_process.is_completed():
                current_process.update_metrics(current_time)
                completed_count += 1
            else:
                # Re-add to queue if not completed
                ready_queue.append(current_process)
