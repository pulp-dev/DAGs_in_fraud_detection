"""Обучение CatBoost на новом окне."""

import logging
import shutil
from datetime import datetime, timezone

import pandas as pd

from pipeline.settings import LABEL, MODELS_DIR

logger = logging.getLogger(__name__)


def prepare_features(frame: pd.DataFrame, cat_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Отделяет target от признаков и проверяет, что категориальные признаки на месте."""
    if LABEL not in frame.columns:
        raise ValueError(f"В датасете нет колонки {LABEL}")
    missing = [column for column in cat_features if column not in frame.columns]
    if missing:
        raise ValueError(f"После препроцессинга нет категориальных признаков {missing}")
    return frame.drop(columns=[LABEL]), frame[LABEL]


def train_model(train: pd.DataFrame, valid: pd.DataFrame, cat_features: list[str]):
    """Обучает CatBoost на train с ранней остановкой по valid."""
    from catboost import CatBoostClassifier

    x_train, y_train = prepare_features(train, cat_features)
    x_valid, y_valid = prepare_features(valid, cat_features)
    model = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        early_stopping_rounds=30,
        random_seed=42,
        allow_writing_files=False,
        verbose=50,
    )
    model.fit(x_train, y_train, cat_features=cat_features, eval_set=(x_valid, y_valid), use_best_model=True)
    return model


def evaluate(model, test: pd.DataFrame, cat_features: list[str]) -> dict:
    """Считает ROC-AUC, PR-AUC, precision, recall и F1 при пороге 0.5 на test."""
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    x_test, y_test = prepare_features(test, cat_features)
    proba = model.predict_proba(x_test[list(model.feature_names_)])[:, 1]
    pred = (proba > 0.5).astype(int)
    two_classes = y_test.nunique() > 1
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_test, proba)) if two_classes else None,
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "fraud_rate": float(y_test.mean()),
    }


def fit_catboost(preproc_info: dict) -> dict:
    """Обучает CatBoost и считает метрики на свежем test-месяце.

    Читает обработанные train, valid, test и stats.json по путям из preproc_info.
    Пишет черновики .staging_<version>.cbm и .staging_<version>.stats.json в каталог моделей.
    latest.txt не меняется, поэтому повтор задачи оставит лишь лишний черновик
    и не затронет скоринг.
    Возвращает версию, пути к черновикам, метрики и метаданные окна.
    """
    cat_features = preproc_info["cat_features"]
    train = pd.read_parquet(preproc_info["train_proc_path"])
    valid = pd.read_parquet(preproc_info["valid_proc_path"])
    test = pd.read_parquet(preproc_info["test_proc_path"])
    model = train_model(train, valid, cat_features)
    metrics = evaluate(model, test, cat_features)
    logger.info("Качество на test %s: %s", preproc_info["test_month"], metrics)

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    staging_model = MODELS_DIR / f".staging_{version}.cbm"
    staging_stats = MODELS_DIR / f".staging_{version}.stats.json"
    model.save_model(str(staging_model))
    shutil.copyfile(preproc_info["stats_path"], staging_stats)
    logger.info("Черновик модели сохранен: %s", staging_model)
    return {
        "version": version,
        "staging_model": str(staging_model),
        "staging_stats": str(staging_stats),
        "dataset_path": preproc_info["dataset_path"],
        "test_raw_path": preproc_info["test_raw_path"],
        "batch_month": preproc_info["batch_month"],
        "train_months": preproc_info["train_months"],
        "test_month": preproc_info["test_month"],
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "best_iteration": int(model.get_best_iteration()),
        **metrics,
    }
