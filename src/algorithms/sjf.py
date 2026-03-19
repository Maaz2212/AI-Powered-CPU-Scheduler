from src.algorithms.scheduler_base import Scheduler

class SJFScheduler(Scheduler):
    def run(self):
        # Sort by arrival time initially to handle the timeline correctly
        self.processes.sort(key=lambda p: p.arrival_time)
        
        n = len(self.processes)
        completed_count = 0
        current_time = 0
        completed = [False] * n
        
        while completed_count < n:
            # Find processes that have arrived and are not completed
            candidates = []
            for i in range(n):
                if self.processes[i].arrival_time <= current_time and not completed[i]:
                    candidates.append(i)
            
            if not candidates:
                # No process available, jump to next arrival
                next_arrival = min(p.arrival_time for p, c in zip(self.processes, completed) if not c)
                current_time = next_arrival
                continue
                
            # Select process with shortest burst time
            # Tie-breaker: FCFS (index in sorted list is already FCFS as we sorted by arrival)
            shortest_idx = min(candidates, key=lambda i: self.processes[i].burst_time)
            process = self.processes[shortest_idx]
            
            # Execute process
            if process.first_run_time == -1:
                process.first_run_time = current_time
            
            process.start_time = current_time

            if process.response_time == -1:
                process.response_time = current_time - process.arrival_time
            
            
            for _ in range(process.burst_time):
                self.log_execution(current_time, process.pid)
                current_time += 1
                
            process.update_metrics(current_time)
            completed[shortest_idx] = True
            completed_count += 1
