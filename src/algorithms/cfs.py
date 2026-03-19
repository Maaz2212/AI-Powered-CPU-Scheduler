"""
CFS — Completely Fair Scheduler (Linux Kernel Simulation)
===========================================================
The Completely Fair Scheduler has been the default Linux process scheduler
since kernel version 2.6.23 (2007). It replaced the older O(1) scheduler.

Core Concept — Virtual Runtime (vruntime):
    Instead of using burst time or priorities directly, CFS tracks how much
    CPU time each process has actually received, normalized by its weight.

    vruntime = actual_runtime × (NICE_0_WEIGHT / process_weight)

    CFS always picks the process with the LOWEST vruntime next.
    This guarantees every process gets a "fair" share of CPU over time.

Key difference from traditional schedulers:
    FCFS/SJF/Priority → decide based on burst time / priority
    CFS               → decides based on who has received LEAST cpu so far

Nice value → Weight mapping (from Linux kernel source):
    nice=-20 → weight=88761 (highest priority)
    nice=0   → weight=1024  (default)
    nice=19  → weight=15    (lowest priority)

References:
    - KernelOracle.pdf (in your repo)
    - https://www.kernel.org/doc/html/latest/scheduler/sched-design-CFS.html
    - Silberschatz OS textbook, Chapter 5
"""

from src.algorithms.scheduler_base import Scheduler
from src.models.process import Process
from typing import Optional

# ── Nice value → weight table (from Linux kernel sched/core.c) ──────────────
NICE_TO_WEIGHT = {
    -20: 88761, -19: 71755, -18: 56483, -17: 46273, -16: 36291,
    -15: 29154, -14: 23254, -13: 18705, -12: 14949, -11: 11916,
    -10:  9548,  -9:  7620,  -8:  6100,  -7:  4904,  -6:  3906,
     -5:  3121,  -4:  2501,  -3:  1991,  -2:  1586,  -1:  1277,
      0:  1024,   1:   820,   2:   655,   3:   526,   4:   423,
      5:   335,   6:   272,   7:   215,   8:   172,   9:   137,
     10:   110,  11:    87,  12:    70,  13:    56,  14:    45,
     15:    36,  16:    29,  17:    23,  18:    18,  19:    15,
}
NICE_0_WEIGHT = 1024   # weight for nice=0 (baseline)


def nice_to_weight(nice: int) -> int:
    """Convert a nice value to its Linux kernel scheduling weight."""
    nice = max(-20, min(19, nice))   # clamp to valid range
    return NICE_TO_WEIGHT.get(nice, NICE_0_WEIGHT)


class CFSScheduler(Scheduler):
    """
    Simulation of the Linux Completely Fair Scheduler.

    At each time unit:
      1. Add all newly arrived processes to the run queue
      2. Pick the process with the lowest vruntime
      3. Run it for 1 time unit
      4. Update its vruntime:
            vruntime += 1 × (NICE_0_WEIGHT / process_weight)
      5. Repeat

    The process with lower nice value (higher priority) has higher weight,
    so its vruntime grows SLOWER → it gets selected more often.
    This is exactly how the real Linux kernel behaves.
    """

    def __init__(self):
        super().__init__()
        # vruntime tracks how much "fair" CPU time each process has received
        self._vruntime: dict[int, float] = {}   # pid → vruntime

    def _get_vruntime(self, pid: int) -> float:
        return self._vruntime.get(pid, 0.0)

    def _update_vruntime(self, process: Process, actual_runtime: int = 1):
        """
        vruntime += actual_runtime × (NICE_0_WEIGHT / weight)

        Process with nice=0  → weight=1024 → vruntime grows at rate 1.0
        Process with nice=-5 → weight=3121 → vruntime grows at rate 0.33
                                              (gets selected ~3× more often)
        Process with nice=5  → weight=335  → vruntime grows at rate 3.06
                                              (gets selected ~3× less often)
        """
        weight   = nice_to_weight(process.priority)  # we use priority as nice
        delta    = actual_runtime * (NICE_0_WEIGHT / weight)
        self._vruntime[process.pid] = self._get_vruntime(process.pid) + delta

    def run(self):
        self.processes.sort(key=lambda p: p.arrival_time)
        n             = len(self.processes)
        current_time  = 0
        completed     = 0
        ready_queue   = []
        process_idx   = 0

        # Initialize vruntime for all processes to 0
        for p in self.processes:
            self._vruntime[p.pid] = 0.0

        while completed < n:
            # Add newly arrived processes to the ready queue
            while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
                ready_queue.append(self.processes[process_idx])
                process_idx += 1

            if not ready_queue:
                # CPU idle — jump to next arrival
                if process_idx < n:
                    current_time = self.processes[process_idx].arrival_time
                continue

            # CFS core: pick process with minimum vruntime
            current_process = min(ready_queue, key=lambda p: self._get_vruntime(p.pid))

            # Track first run time (response time)
            if current_process.first_run_time == -1:
                current_process.first_run_time = current_time
            if current_process.start_time == -1:
                current_process.start_time = current_time

            # Run for 1 time unit
            self.log_execution(current_time, current_process.pid)
            self._update_vruntime(current_process, actual_runtime=1)
            current_process.remaining_time -= 1
            current_time += 1

            # Check for new arrivals during this tick
            while process_idx < n and self.processes[process_idx].arrival_time <= current_time:
                ready_queue.append(self.processes[process_idx])
                process_idx += 1

            # Check completion
            if current_process.is_completed():
                current_process.update_metrics(current_time)
                ready_queue.remove(current_process)
                completed += 1