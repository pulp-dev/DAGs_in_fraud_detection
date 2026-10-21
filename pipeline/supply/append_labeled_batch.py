"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from pipeline.db import get_conn, replace_labeled_month

logger = logging.getLogger(__name__)


def append_labeled_batch(batch_info: dict) -> dict:
    """Записывает parquet батча в labeled_transactions как месяц batch_month.

    Читает путь и месяц из batch_info. Перед загрузкой удаляет строки этого
    месяца в той же транзакции. Месяц батча идет после максимальной даты витрины,
    поэтому удаляются только остатки прошлой попытки этой же задачи:
    повтор после сбоя не задваивает батч. Строки других месяцев не меняются.
    Возвращает месяц и число записанных строк.
    """
    batch_path = Path(batch_info["batch_path"])
    if not batch_path.exists():
        raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")

    frame = pd.read_parquet(batch_path)
    month_start = date.fromisoformat(batch_info["batch_month"])
    conn = get_conn()
    try:
        replace_labeled_month(conn, frame, month_start)
    finally:
        conn.close()
    logger.info("Месяц %s добавлен в витрину, строк: %s", month_start, len(frame))
    return {
        "batch_path": batch_info["batch_path"],
        "batch_month": batch_info["batch_month"],
        "rows": int(len(frame)),
    }
