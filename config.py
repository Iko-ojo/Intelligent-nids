"""
Configuration settings for Network Anomaly Detection project.

This module contains all configuration parameters including data paths,
model hyperparameters, and training settings.
"""
import os
from pathlib import Path

# Project root directory
# `config.py` lives at the repository root, so `parent` is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Data configuration
# -----------------------------------------------------------------------------
# Base directory for all external datasets
DATA_ROOT = PROJECT_ROOT / "new-dt"

# CIC‑UNSW dataset lives under `new-dt/CIC-UNSW/` and is split into:
#   - `Data.csv`  -> feature matrix
#   - `Label.csv` -> labels (single "Label" column)
CIC_UNSW_DIR = DATA_ROOT / "CIC-UNSW"
FEATURES_FILE = CIC_UNSW_DIR / "Data.csv"
LABELS_FILE = CIC_UNSW_DIR / "Label.csv"

# When using the new CIC‑UNSW split files we no longer rely on a single
# merged CSV. Keep this here for backwards compatibility – if you want to use
# the old merged file, point DATASET_FILE to it and set FEATURES_FILE/LABELS_FILE
# to None.
DATASET_FILE = None  # e.g. DATA_ROOT / "merged_150k_70_30.csv"

# For older workflows that used explicit train/test CSVs. Leave as None for
# the new CIC‑UNSW setup.
TRAIN_FILE = None
TEST_FILE = None

# Label column name
LABEL_COLUMN = "Label"

MULTICLASS_LABELS = {
    0: "BENIGN",
    1: "Web Attacks",
    2: "DDOS",
    3: "PortScan",
}

# Model training parameters
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2  # For train/test split if needed

# Machine Learning model parameters
ML_MODELS_CONFIG = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  # Use all available cores
}

# XGBoost specific parameters (num_class=4 for BENIGN, Web, DDoS, PortScan)
# Note: random_state and n_jobs are passed by create_ml_models, not here
XGBOOST_PARAMS = {
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "num_class": 4,
}

# Deep Learning parameters
DL_CONFIG = {
    "cnn": {
        "epochs": 20,
        "batch_size": 256,
        "learning_rate": 0.001,
        "validation_split": VALIDATION_SPLIT,
    },
    "rnn": {
        "epochs": 20,
        "batch_size": 256,
        "learning_rate": 0.0005,
        "validation_split": VALIDATION_SPLIT,
    },
    "ann": {
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "validation_split": VALIDATION_SPLIT,
    },
    "mlp": {
        "epochs": 50,
        "batch_size": 512,
        "learning_rate": 0.0001,
        "validation_split": VALIDATION_SPLIT,
    },
}

# Early stopping configuration
EARLY_STOPPING_CONFIG = {
    "monitor": "val_loss",
    "patience": 5,
    "restore_best_weights": True,
    "verbose": 1,
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "style": "seaborn-v0_8",
    "figure_size_large": (20, 15),
    "figure_size_medium": (12, 8),
    "figure_size_small": (8, 6),
    "dpi": 100,
    "color_palette": "husl",
}

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Target names for classification (must match the order above)
TARGET_NAMES = [
    MULTICLASS_LABELS[0],
    MULTICLASS_LABELS[1],
    MULTICLASS_LABELS[2],
    MULTICLASS_LABELS[3],
]

# Logging configuration
LOG_LEVEL = "INFO"
VERBOSE = True

