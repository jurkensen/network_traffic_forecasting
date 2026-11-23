# Network Traffic Forecasting

Bachelor thesis project from TU Delft comparing forecasting models for mobile network traffic. The project evaluates interval aggregation and compares ARIMA/SARIMA, RNN, and LSTM models on three base-station datasets.

## Models

- ARIMA/SARIMA baseline
- Simple RNN
- LSTM
- Nasseri LSTM
- TimesFM

## Setup

```bash
python -m pip install -r requirements.txt
```

Python 3.10+ is recommended. GPU training may additionally require a CUDA-enabled PyTorch installation.

## Run

Run the default SARIMA interval study:

```bash
python -m main
```

Run a neural-network interval study:

```bash
python -m models.RNN
python -m models.LSTM
```

The main SARIMA study uses the PobleSec training and test CSVs. It evaluates 5-, 15-, 30-, and 60-minute intervals with 1-day, 1-week, and 2-week training windows.

## Data

Datasets are stored in `datasets/`:

- `Elborn`
- `LesCorts`
- `PobleSec`

Each station directory contains a training CSV and a test CSV. Data loading and preprocessing utilities are in `utils/preprocessing.py`.

## Project Layout

- `main.py` - SARIMA interval-study entry point
- `models/` - forecasting implementations and training helpers
- `utils/` - configuration, preprocessing, plotting, and cross-validation utilities
- `notebooks/` - data analysis and TimesFM experiments
- `scripts/` - example SLURM training jobs
- `experiments/` and `general/` - saved experiment outputs and plots
