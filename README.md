# AI-Powered CPU Scheduling Simulator

## Overview
This project is an advanced CPU Process Scheduling Simulator built with **Flask (Python)**. It allows users to visualize and compare classical CPU scheduling algorithms through a web-based interface. 

What sets this simulator apart is its **Machine Learning pipeline**. It has been specifically trained on over **122,000 empirical operating system snapshots** to predict true CPU Burst Times using `RandomForestRegressor`, `GradientBoostingRegressor`, and Exponential Weighted Moving Averages (EWMA) synchronously. By stripping away stochastic OS noise, the models achieve up to `0.999` mathematically verified Test R² scores using strictly the core structural features of a process.

## Features
- **Algorithms Supported:**
  - FCFS (First Come First Serve)
  - SJF (Shortest Job First - Non-preemptive)
  - SRTF (Shortest Remaining Time First - Preemptive SJF)
  - Priority Scheduling (Preemptive & Non-preemptive)
  - CFS (Completely Fair Scheduler - Weight based)
  - Round Robin (Configurable Time Quantum)
- **Synchronous ML Prediction Workflow:** Input a process's hardware telemetry and watch 3 separate AI/Heuristic models simulate and predict its Burst Time instantly side-by-side via a dynamic comparative UI.
- **Dimension Reduction:** Models are rigorously hyper-optimized to predict burst times accurately using only **4 deterministic features**: Voluntary Context Switches, Memory %, I/O Read Bytes, and Num Threads. 
- **Visualization:** Generates dynamic HTML/CSS Gantt charts for process execution without heavy canvas libraries.

## Tech Stack
- **Backend:** Python, Flask, Pandas, Scikit-Learn (Random Forest, Gradient Boosting)
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla DOM rendering)
- **Logic:** Custom Python implementations of ML metrics parsing and core scheduling algorithms.

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

## API Endpoints (Machine Learning)

- **Predict Burst Time (Multi-Model Synchronous):**
  - `POST /predict-burst`
    - Evaluates CPU burst time computationally across `Random Forest`, `Gradient Boosting`, and `EMMA`.
    - Request body:
      ```json
      {
        "name": "chrome.exe", 
        "memory_percent": 0.32, 
        "num_threads": 20, 
        "io_read_bytes": 1390784, 
        "num_ctx_switches_voluntary": 27499 
      }
      ```
    - Response:
      ```json
      { 
        "rf": 1.352, "gb": 0.942, "emma": 1.250, "unit": "seconds" 
      }
      ```

- **Retrain System Database:**
  - `POST /train-all`
    - Reads `process_data.csv`, shuffles a strict 70-30 Train/Test split to prevent data leakage, refits the ML pipelines, and returns rigorous MAE/R² evaluation metrics.

- **Model Info:**
  - `GET /model-info`
    - Returns operational metadata about the predictors and empirical telemetry features.

## Project Structure
```text
/
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── src/
│   ├── algorithms/         # Classical scheduling algorithm functions
│   ├── ml/                 # Machine learning environment
│   │   ├── process_data.csv        # 122k row empirical telemetry dataset
│   │   ├── rf_predictor.py         # Random Forest pipeline & logic
│   │   ├── gb_predictor.py         # Gradient Boosting pipeline & logic
│   │   ├── rf_model.pkl            # Persistent RF parameters
│   │   └── gb_model.pkl            # Persistent GB parameters
│   ├── models/             # OOP Data structures (e.g. Process)
│   └── utils/              # Evaluation & helper utilities
├── static/
│   └── style.css           # UI layout and Grid/Flex styling
└── templates/
    └── index.html          # Main comparative Dashboard & Simulator bounds
```

---
*Created as part of the AI-Powered CPU Scheduling architectural overhaul.*