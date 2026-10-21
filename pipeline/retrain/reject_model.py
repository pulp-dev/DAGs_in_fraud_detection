"""Сохранение модели, не прошедшей quality gate."""

import logging

from pipeline.retrain.artifacts import store_artifacts
from pipeline.settings import REJECTED_DIR

logger = logging.getLogger(__name__)


def reject_model(gate_info: dict) -> str:
    """Откладывает новую модель в models/rejected, не трогая прод.

    Переносит черновики в rejected/my_catboost_<version>.cbm и .stats.json
    и пишет metrics.json с метриками обеих моделей и причиной отказа.
    latest.txt не меняется, скоринг продолжает работать на прежней версии.
    Повтор безопасен: перенос пропускается, если файлы уже на месте.
    Возвращает версию отклоненной модели.
    """
    final_model = store_artifacts(gate_info, REJECTED_DIR, "rejected")
    logger.info("Модель %s не опубликована, сохранена в %s", gate_info["version"], final_model)
    return gate_info["version"]
