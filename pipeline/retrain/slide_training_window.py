"""Сдвиг окна обучения на один месяц вперед."""

import logging
from datetime import date

import pandas as pd

from pipeline.db import get_conn, read_labeled_range, window_start_for
from pipeline.settings import PENDING_WINDOW, WINDOW_MONTHS

logger = logging.getLogger(__name__)


def slide_training_window(batch_info: dict) -> dict:
    """Собирает окно из WINDOW_MONTHS месяцев, последний из которых batch_month.

    Читает из labeled_transactions месяцы окна до batch_month и parquet
    батча из batch_info. Пишет pending_window.parquet, current.parquet не трогает.
    Границы окна считаются только от batch_month, а не от прошлого окна на диске.
    Поэтому повтор задачи или всего прогона после сбоя собирает то же самое окно
    и не может сдвинуть его второй раз или задвоить месяц.
    Возвращает путь к новому окну и месяц добавленного батча.
    """
    batch_month = date.fromisoformat(batch_info["batch_month"])
    window_start = window_start_for(batch_month)
    conn = get_conn()
    try:
        history = read_labeled_range(conn, window_start, batch_month)
    finally:
        conn.close()
    batch = pd.read_parquet(batch_info["batch_path"])

    window = pd.concat([history, batch], ignore_index=True)
    window["transaction_time"] = pd.to_datetime(window["transaction_time"])
    window = window.sort_values("transaction_time", kind="stable")
    window.to_parquet(PENDING_WINDOW, index=False)

    months = sorted(window["transaction_time"].dt.to_period("M").unique().astype(str))
    if len(months) < WINDOW_MONTHS:
        logger.warning("В окне %s месяцев вместо %s: %s", len(months), WINDOW_MONTHS, months)
    logger.info("Окно %s: %s строк.", months, len(window))
    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_month.isoformat(),
    }
