"""Фиксация нового окна обучения после сохранения модели."""

import logging
import os
from datetime import date
from pathlib import Path

from pipeline.db import advance_training_state, get_conn
from pipeline.settings import CURRENT_DATASET

logger = logging.getLogger(__name__)


def advance_window(gate_info: dict) -> str:
    """Сдвигает указатель месяца после того, как модель сохранена в любой из веток gate.

    Запускается после publish_model или reject_model, поэтому указатель никогда
    не обгоняет сохранение модели. Сохраняет окно в current.parquet для истории
    и записывает training_state значениями, вычисленными из batch_month.
    Повтор безопасен: повторная запись того же месяца ничего не меняет,
    а следующий прогон строит окно от training_state, а не от current.parquet.
    Возвращает месяц, которым теперь заканчивается окно.
    """
    pending_window = Path(gate_info["dataset_path"])
    if pending_window.exists():
        os.replace(pending_window, CURRENT_DATASET)

    batch_month = date.fromisoformat(gate_info["batch_month"])
    conn = get_conn()
    try:
        advance_training_state(conn, batch_month)
    finally:
        conn.close()
    return batch_month.isoformat()
