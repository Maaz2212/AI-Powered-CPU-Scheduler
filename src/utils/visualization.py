import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import base64
from typing import List, Dict, Any

def get_gantt_chart_base64(execution_log: List[Dict[str, Any]], title: str = "Gantt Chart") -> str:
    if not execution_log:
        return ""

    # Extract data
    times = [entry['time'] for entry in execution_log]
    pids = [entry['pid'] for entry in execution_log]
    
    unique_pids = sorted(list(set(pids)))
    colors = list(mcolors.TABLEAU_COLORS.values())
    pid_color_map = {pid: colors[i % len(colors)] for i, pid in enumerate(unique_pids)}
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    blocks = []
    if execution_log:
        sorted_log = sorted(execution_log, key=lambda x: x['time'])
        current_pid = sorted_log[0]['pid']
        start_time = sorted_log[0]['time']
        
        for i in range(1, len(sorted_log)):
            next_time = sorted_log[i]['time']
            next_pid = sorted_log[i]['pid']
            
            if next_pid != current_pid or next_time != sorted_log[i-1]['time'] + 1:
                duration = sorted_log[i-1]['time'] - start_time + 1
                blocks.append((start_time, duration, current_pid))
                current_pid = next_pid
                start_time = next_time
        
        duration = sorted_log[-1]['time'] - start_time + 1
        blocks.append((start_time, duration, current_pid))
        
    for start, duration, pid in blocks:
        ax.barh(y=f"P{pid}", width=duration, left=start, color=pid_color_map[pid], edgecolor='black')
        ax.text(start + duration/2, f"P{pid}", f"P{pid}", ha='center', va='center', color='white', fontweight='bold')

    ax.set_xlabel("Time")
    ax.set_ylabel("Process ID")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


