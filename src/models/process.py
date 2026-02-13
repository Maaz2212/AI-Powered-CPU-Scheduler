from dataclasses import dataclass, field

@dataclass
class Process:
    pid: int
    arrival_time: int
    burst_time: int
    priority: int
    remaining_time: int = field(init=False)
    waiting_time: int = 0
    turnaround_time: int = 0
    response_time: int = -1
    completion_time: int = 0
    first_run_time: int = -1
    start_time: int = -1  # Time when process actually starts execution

    def __post_init__(self):
        self.remaining_time = self.burst_time

    def is_completed(self) -> bool:
        return self.remaining_time == 0

    def update_metrics(self, current_time: int):
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time
        self.waiting_time = self.turnaround_time - self.burst_time
