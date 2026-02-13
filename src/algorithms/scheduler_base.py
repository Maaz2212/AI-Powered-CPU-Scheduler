from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.models.process import Process

class Scheduler(ABC):
    def __init__(self):
        self.processes: List[Process] = []
        self.execution_log: List[Dict[str, Any]] = []  # To store Gantt chart data

    def add_process(self, process: Process):
        self.processes.append(process)

    def get_processes(self) -> List[Process]:
        return self.processes
    
    def log_execution(self, time: int, process_id: int):
        """Logs the execution of a process at a specific time."""
        self.execution_log.append({"time": time, "pid": process_id})

    @abstractmethod
    def run(self):
        """Runs the scheduling algorithm."""
        pass

    def get_timeline(self) -> List[Dict[str, Any]]:
        return self.execution_log
