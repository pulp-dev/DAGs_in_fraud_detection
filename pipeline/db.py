"""Подключение к витрине Postgres и состояние окна обучения."""

import io
import logging
import os
from datetime import date

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta

from pipeline.settings import WINDOW_MONTHS

logger = logging.getLogger(__name__)

LABELED_COLUMNS = [
    "transaction_time",
    "merch",
    "cat_id",
    "amount",
    "name_1",
    "name_2",
    "gender",
    "street",
    "one_city",
    "us_state",
    "post_code",
    "lat",
    "lon",
    "population_city",
    "jobs",
    "merchant_lat",
    "merchant_lon",
    "target",
]


def get_conn():
    """Открывает соединение с базой, где лежит витрина labeled_transactions.

    Хост, порт, имя базы и учетные данные читаются из переменных окружения
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD.
    """
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def month_end_exclusive(month_start: date) -> date:
    """Возвращает первое число месяца, следующего за month_start.

    Верхняя граница не включается в выборку месяца, который начинается в month_start.
    """
    return month_start + relativedelta(months=1)


def read_training_state(conn) -> dict:
    """Читает границы трехмесячного окна и месяц следующего батча обучения.

    Ожидает строку id = 1 в training_state, которую создает data_loader.
    Возвращает словарь с ключами window_start, window_end и next_batch_month.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT window_start, window_end, next_batch_month
            FROM training_state
            WHERE id = 1
            """
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError("Таблица training_state пуста. Сначала должен отработать data_loader.")
    return {
        "window_start": row[0],
        "window_end": row[1],
        "next_batch_month": row[2],
    }


def window_start_for(batch_month: date) -> date:
    """Возвращает первый месяц окна из WINDOW_MONTHS месяцев, которое заканчивается batch_month."""
    return batch_month - relativedelta(months=WINDOW_MONTHS - 1)


def read_labeled_range(conn, start: date, end_exclusive: date) -> pd.DataFrame:
    """Читает строки labeled_transactions с transaction_time в [start, end_exclusive)."""
    columns = ", ".join(LABELED_COLUMNS)
    return pd.read_sql(
        f"""
        SELECT {columns}
        FROM labeled_transactions
        WHERE transaction_time >= %s AND transaction_time < %s
        ORDER BY transaction_time
        """,
        conn,
        params=(start, end_exclusive),
    )


def advance_training_state(conn, batch_month: date) -> None:
    """Сдвигает окно обучения на месяц успешно опубликованного батча.

    Новое окно заканчивается batch_month и покрывает WINDOW_MONTHS месяцев.
    next_batch_month становится следующим календарным месяцем.
    Значения вычисляются только из batch_month, поэтому повторный вызов
    с тем же месяцем ничего не меняет.
    """
    window_end = batch_month
    window_start = window_start_for(batch_month)
    next_batch = month_end_exclusive(batch_month)
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE training_state
            SET window_start = %s,
                window_end = %s,
                next_batch_month = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (window_start, window_end, next_batch),
        )
    conn.commit()
    logger.info(
        "Состояние обновлено: окно %s - %s, следующий батч %s",
        window_start,
        window_end,
        next_batch,
    )


def replace_labeled_month(conn, frame: pd.DataFrame, month_start: date) -> None:
    """Записывает frame в labeled_transactions как содержимое месяца month_start.

    Сначала удаляет строки этого месяца, затем загружает frame через COPY.
    Оба шага идут в одной транзакции, поэтому повтор после сбоя
    не создает дублей и не оставляет месяц наполовину записанным.
    """
    payload = frame.loc[:, LABELED_COLUMNS].copy()
    payload["transaction_time"] = pd.to_datetime(payload["transaction_time"])
    buffer = io.StringIO()
    payload.to_csv(
        buffer,
        index=False,
        header=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    buffer.seek(0)
    column_list = ", ".join(LABELED_COLUMNS)
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM labeled_transactions WHERE transaction_time >= %s AND transaction_time < %s",
            (month_start, month_end_exclusive(month_start)),
        )
        deleted = cur.rowcount
        cur.copy_expert(
            f"COPY labeled_transactions ({column_list}) FROM STDIN WITH (FORMAT CSV)",
            buffer,
        )
    conn.commit()
    logger.info(
        "Месяц %s в labeled_transactions: удалено %s строк, записано %s.",
        month_start,
        deleted,
        len(payload),
    )
