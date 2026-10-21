"""Публикация модели, прошедшей quality gate."""

import logging
import os

from pipeline.retrain.artifacts import store_artifacts
from pipeline.settings import MODELS_DIR

logger = logging.getLogger(__name__)


def publish_model(gate_info: dict) -> str:
    """Выкладывает новую модель в прод.

    Переносит черновики в my_catboost_<version>.cbm и .stats.json, пишет metrics.json
    с решением gate и только после этого атомарно обновляет latest.txt,
    поэтому скоринг не увидит версию без статистик.
    Повтор безопасен: перенос пропускается, если файлы уже на месте,
    а latest.txt перезаписывается той же версией.
    Возвращает опубликованную версию.
    """
    version = gate_info["version"]
    final_model = store_artifacts(gate_info, MODELS_DIR, "published")
    latest_tmp = MODELS_DIR / f".latest.{version}.txt"
    latest_tmp.write_text(version + "\n", encoding="utf-8")
    os.replace(latest_tmp, MODELS_DIR / "latest.txt")
    logger.info("Опубликована модель %s (%s)", version, final_model)
    return version
