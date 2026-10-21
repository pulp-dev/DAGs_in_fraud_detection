"""Оценка текущей prod-модели на том же test-месяце, что и новая модель."""

import json
import logging

import pandas as pd

from pipeline.retrain.fit_catboost import evaluate
from pipeline.retrain.preprocess_dataset import load_preprocessing
from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def read_champion_version() -> str | None:
    """Возвращает версию из latest.txt или None, если prod-модели еще нет."""
    latest = MODELS_DIR / "latest.txt"
    if not latest.exists():
        return None
    return latest.read_text(encoding="utf-8").strip() or None


def evaluate_champion(train_info: dict) -> dict:
    """Считает метрики prod-модели на свежем test-месяце новой модели.

    Читает latest.txt, my_catboost_<version>.cbm и .stats.json текущей prod-версии
    и test_raw.parquet по пути из train_info. Признаки строятся статистиками самой
    prod-модели, как в онлайн-скоринге. Ничего не пишет, поэтому повтор безопасен.
    Возвращает train_info с полями champion_version и champion_metrics
    (None, если prod-модели нет или ее артефакты не найдены).
    """
    from catboost import CatBoostClassifier

    version = read_champion_version()
    result = {**train_info, "champion_version": version, "champion_metrics": None}
    if version is None:
        logger.info("Prod-модели нет, сравнивать не с чем.")
        return result

    model_path = MODELS_DIR / f"my_catboost_{version}.cbm"
    stats_path = MODELS_DIR / f"my_catboost_{version}.stats.json"
    if not model_path.exists() or not stats_path.exists():
        logger.warning("Артефакты prod-модели %s не найдены, сравнивать не с чем.", version)
        return result

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    preprocessing = load_preprocessing()
    test = preprocessing.run_preproc(pd.read_parquet(train_info["test_raw_path"]), stats)
    result["champion_metrics"] = evaluate(model, test, preprocessing.CAT_FEATURES)
    logger.info(
        "Test %s: prod %s %s, новая модель %s",
        train_info["test_month"],
        version,
        result["champion_metrics"],
        {key: train_info[key] for key in ("roc_auc", "pr_auc", "f1")},
    )
    return result
