"""Очистка окна обучения от невалидных строк."""

import logging

import pandas as pd

from pipeline.settings import CLEAN_WINDOW, LABEL

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["transaction_time", "amount", "cat_id", "lat", "lon", "merchant_lat", "merchant_lon", LABEL]


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Удаляет строки, на которых нельзя учиться, и возвращает число удаленных по каждому правилу.

    Правила применяются последовательно: пропуски в обязательных колонках,
    target вне {0, 1}, amount <= 0, координаты вне допустимых диапазонов, полные дубликаты.
    """
    report = {}

    def drop(name: str, mask: pd.Series) -> None:
        nonlocal frame
        report[name] = int(mask.sum())
        frame = frame.loc[~mask]

    drop("missing_required", frame[REQUIRED_COLUMNS].isna().any(axis=1))
    drop("bad_target", ~frame[LABEL].isin([0, 1]))
    drop("non_positive_amount", frame["amount"] <= 0)
    bad_coords = pd.Series(False, index=frame.index)
    for lat, lon in (("lat", "lon"), ("merchant_lat", "merchant_lon")):
        bad_coords |= ~frame[lat].between(-90, 90) | ~frame[lon].between(-180, 180)
    drop("bad_coordinates", bad_coords)
    drop("duplicates", frame.duplicated())
    frame = frame.astype({LABEL: int}).reset_index(drop=True)
    return frame, report


def clean_dataset(window_info: dict) -> dict:
    """Чистит окно обучения перед split.

    Читает pending_window.parquet по пути из window_info, пишет clean_window.parquet.
    Входной файл не меняется, а выходной перезаписывается целиком,
    поэтому повтор задачи дает тот же результат.
    Возвращает путь к очищенному окну, месяц батча и число удаленных строк по правилам.
    """
    frame = pd.read_parquet(window_info["dataset_path"])
    cleaned, report = clean_frame(frame)
    if cleaned.empty:
        raise RuntimeError("После очистки окно обучения пустое.")
    cleaned.to_parquet(CLEAN_WINDOW, index=False)
    logger.info("Очистка окна: было %s, стало %s строк, удалено %s", len(frame), len(cleaned), report)
    return {
        "dataset_path": window_info["dataset_path"],
        "clean_path": str(CLEAN_WINDOW),
        "batch_month": window_info["batch_month"],
        "dropped": report,
    }
