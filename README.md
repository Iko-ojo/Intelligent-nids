# Intelligent-nids

This project contains `model_comparison.ipynb`, a unified experiment notebook for network anomaly detection model benchmarking.

## What the notebook does

- Loads and preprocesses CIC-UNSW data.
- Trains and compares six models:
  - Isolation Forest
  - Random Forest
  - XGBoost
  - KMeans
  - Hybrid IF + RF
  - Hybrid XGBoost + KMeans
- Evaluates model performance using common classification metrics.
- Generates comparison visualizations and saves outputs to the project output directories.

## Dataset notice

The dataset files are **not included in this repository** due to GitHub file size restrictions.

Please download the required UNSW-NB15/CIC dataset files from the official UNB CIC datasets page:

- [UNB CIC Datasets](https://www.unb.ca/cic/datasets/index.html)

After downloading, place the dataset files in the paths expected by `config.py` (for example under `new-dt/CIC-UNSW/`).

## Running the notebook

1. Create/activate your Python environment.
2. Install project dependencies.
3. Ensure dataset files are in the configured location.
4. Open and run `model_comparison.ipynb` from the repository root (or a path where `config.py` can be resolved).
