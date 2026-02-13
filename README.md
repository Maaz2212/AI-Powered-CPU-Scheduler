# CPU Process Scheduling Simulator

## Overview
This project is a CPU Process Scheduling Simulator built with **Flask (Python)**. It allows users to visualize and compare different CPU scheduling algorithms through a web-based interface.

## Features
- **Algorithms Supported:**
  - FCFS (First Come First Serve)
  - SJF (Shortest Job First - Non-preemptive)
  - SRTF (Shortest Remaining Time First - Preemptive SJF)
  - Priority Scheduling (Preemptive & Non-preemptive)
  - Round Robin (Configurable Time Quantum)
- **metrics:** Calculates Average Waiting Time, Turnaround Time, CPU Utilization, Throughput, and Context Switches.
- **Visualization:** Generates dynamic Gantt charts for process execution.
- **Dynamic UI:** Add/Remove processes, configure algorithms, and view results on a single page.

## Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Visualization:** Matplotlib
- **Logic:** Custom Python implementations of scheduling algorithms

## Installation Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <project-folder>
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate   # Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Start the Flask Application:**
   ```bash
   python3 app.py
   ```

2. **Open in Browser:**
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Project Structure
```
/
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Project dependencies
├── src/
│   ├── algorithms/         # Scheduling algorithm implementations
│   ├── models/             # Data models (Process class)
│   └── utils/              # Utilities for metrics and visualization
├── static/
│   └── style.css           # CSS styles
└── templates/
    └── index.html          # Main HTML template (Single Page App)
```