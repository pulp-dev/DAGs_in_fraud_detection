"""Извлечение очередного месяца из размеченной витрины."""

import logging

from airflow.exceptions import AirflowSkipException

from pipeline.db import get_conn, month_end_exclusive, read_labeled_range, read_training_state
from pipeline.settings import BATCHES_DIR

logger = logging.getLogger(__name__)


def extract_month_batch() -> dict:
    """Забирает из labeled_transactions месяц, указанный в training_state.

    Читает next_batch_month и строки этого календарного месяца из витрины.
    Пишет batches/batch_<YYYYMM>.parquet. Указатель месяца не меняет, а файл
    батча перезаписывается целиком, поэтому повтор дает тот же результат.
    Если строк нет, помечает задачу как skipped: новый месяц еще не поставлен.
    Возвращает путь к файлу батча и месяц в формате ISO.
    """
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    try:
        state = read_training_state(conn)
        batch_month = state["next_batch_month"]
        logger.info(
            "Текущее окно %s - %s. Извлекаем месяц %s.",
            state["window_start"],
            state["window_end"],
            batch_month,
        )
        frame = read_labeled_range(conn, batch_month, month_end_exclusive(batch_month))
    finally:
        conn.close()

    if frame.empty:
        raise AirflowSkipException(
            f"В витрине нет наблюдений за {batch_month}. Новый месяц еще не поставлен."
        )

    batch_path = BATCHES_DIR / f"batch_{batch_month.strftime('%Y%m')}.parquet"
    frame.to_parquet(batch_path, index=False)
    logger.info(
        "Батч %s: %s строк, %s - %s",
        batch_month,
        len(frame),
        frame["transaction_time"].min(),
        frame["transaction_time"].max(),
    )
    return {
        "batch_path": str(batch_path),
        "batch_month": batch_month.isoformat(),
    }
