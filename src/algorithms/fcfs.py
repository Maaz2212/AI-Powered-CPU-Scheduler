from src.algorithms.scheduler_base import Scheduler

class FCFSScheduler(Scheduler):
    def run(self):
        # Sort processes by arrival time
        self.processes.sort(key=lambda p: p.arrival_time)
        
        current_time = 0
        for process in self.processes:
            if current_time < process.arrival_time:
                current_time = process.arrival_time
            
            # First run time is when the process starts execution
            if process.first_run_time == -1:
                process.first_run_time = current_time
            
            process.start_time = current_time
            
            # Log execution
            # Log every unit of time for granular Gantt chart or just start/end?
            # For simplicity in Gantt, let's log the duration block or unit by unit.
            # Logging unit by unit allows for easier preemption visualization later.
            for _ in range(process.burst_time):
                self.log_execution(current_time, process.pid)
                current_time += 1
            
            process.update_metrics(current_time)
