"""Общие пути артефактов обучения и поставки батчей."""

import os
from pathlib import Path

TRAINING_DIR = Path(os.getenv("TRAINING_DATA_DIR", "/opt/airflow/training_data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/opt/airflow/models"))
FRAUD_SRC = os.getenv("FRAUD_SRC", "/opt/airflow/fraud_src")
DQ_REPORTS_DIR = Path(os.getenv("DQ_REPORTS_DIR", "/opt/airflow/dq_reports"))

CURRENT_DATASET = TRAINING_DIR / "current.parquet"
PENDING_WINDOW = TRAINING_DIR / "pending_window.parquet"
CLEAN_WINDOW = TRAINING_DIR / "clean_window.parquet"
BATCHES_DIR = TRAINING_DIR / "batches"
SPLITS_DIR = TRAINING_DIR / "splits"
SUPPLY_BATCH = TRAINING_DIR / "supply_batch.parquet"
REJECTED_DIR = MODELS_DIR / "rejected"

LABEL = "target"
SAMPLE_SIZE = 50_000
HISTORY_YEARS = 2
WINDOW_MONTHS = 3
VALID_FRACTION = 0.2
GATE_METRIC = "pr_auc"
GATE_TOLERANCE = 0.0
