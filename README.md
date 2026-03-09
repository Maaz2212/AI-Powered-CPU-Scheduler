# CPU Process Scheduling Simulator

## Overview
This project is a CPU Process Scheduling Simulator built with **Flask (Python)**. It allows users to visualize and compare different CPU scheduling algorithms through a web-based interface. It also includes a machine learning module for predicting CPU burst times using Support Vector Regression (SVR).

## Features
- **Algorithms Supported:**
  - FCFS (First Come First Serve)
  - SJF (Shortest Job First - Non-preemptive)
  - SRTF (Shortest Remaining Time First - Preemptive SJF)
  - Priority Scheduling (Preemptive & Non-preemptive)
  - Round Robin (Configurable Time Quantum)
- **Metrics:** Calculates Average Waiting Time, Turnaround Time, CPU Utilization, Throughput, and Context Switches.
- **Visualization:** Generates dynamic Gantt charts for process execution.
- **Dynamic UI:** Add/Remove processes, configure algorithms, and view results on a single page.
- **Machine Learning (NEW):** Predict CPU burst time for processes using an SVM model trained on real process data. Includes endpoints for prediction and retraining.

## Tech Stack
- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **Visualization:** Matplotlib
- **Machine Learning:** scikit-learn, pandas, numpy
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

## API Endpoints (Machine Learning)

- **Predict Burst Time:**
  - `POST /predict-burst`
    - Predicts CPU burst time for one or more processes using the trained SVM model.
    - Request body (single):
      ```json
      { "memory_percent": 1.5, "num_threads": 4, ... }
      ```
    - Request body (batch):
      ```json
      { "processes": [ { ... }, { ... } ] }
      ```
    - Response:
      ```json
      { "predicted_burst_time": 7.23, "unit": "seconds", "model": "SVM (SVR, RBF kernel)" }
      ```

- **Retrain SVM Model:**
  - `POST /train-svm`
    - Retrains the SVM model on the latest `process_data.csv` and returns evaluation metrics.

- **SVM Model Info:**
  - `GET /svm-info`
    - Returns metadata about the SVM predictor, features, and training data.

## Project Structure
```
/
├── app.py                  # Main Flask application entry point
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── src/
│   ├── algorithms/         # Scheduling algorithm implementations
│   │   ├── fcfs.py
│   │   ├── sjf.py
│   │   ├── srtf.py
│   │   ├── priority.py
│   │   ├── round_robin.py
│   │   └── scheduler_base.py
│   ├── models/             # Data models (Process class)
│   │   └── process.py
│   ├── utils/              # Utilities for metrics and visualization
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── ml/                 # Machine learning module (SVM burst time prediction)
│       ├── process_data.csv    # Real process data for training
│       ├── svm_model.pkl       # Trained SVM model (auto-generated)
│       └── svm_predictor.py    # SVM training and prediction logic
├── static/
│   └── style.css           # CSS styles
└── templates/
    └── index.html          # Main HTML template (Single Page App)
```

## Data & Model Files
- `src/ml/process_data.csv`: Real process snapshots used for SVM training.
- `src/ml/svm_model.pkl`: Trained SVM model (auto-generated after training).

## Notes
- The SVM model is automatically trained if not found when a prediction is requested.
- You can retrain the model anytime by calling the `/train-svm` endpoint.
- All scheduling logic and ML code is modular and easy to extend.

---
For more details, see code comments and docstrings in the source files.