"""Временной split окна на train, valid и test."""

import logging

import pandas as pd

from pipeline.settings import SPLITS_DIR, VALID_FRACTION

logger = logging.getLogger(__name__)


def split_by_time(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Свежий месяц уходит в test, ранние месяцы по времени делятся на train и valid.

    valid это последние VALID_FRACTION строк ранних месяцев, он нужен для ранней
    остановки, чтобы test-месяц не влиял на обучение.
    """
    frame = frame.copy()
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    frame = frame.sort_values("transaction_time", kind="stable")
    periods = frame["transaction_time"].dt.to_period("M")
    latest = periods.max()
    history = frame.loc[periods < latest]
    test = frame.loc[periods == latest]
    cut = int(len(history) * (1 - VALID_FRACTION))
    train, valid = history.iloc[:cut], history.iloc[cut:]
    if train.empty or valid.empty or test.empty:
        raise RuntimeError(
            f"Не удалось разрезать окно по времени: train={len(train)}, "
            f"valid={len(valid)}, test={len(test)}"
        )
    return train, valid, test


def time_train_test_split(clean_info: dict) -> dict:
    """Делит очищенное окно по времени на train, valid и test.

    Читает clean_window.parquet по пути из clean_info.
    Пишет splits/train_raw.parquet, splits/valid_raw.parquet, splits/test_raw.parquet.
    Файлы перезаписываются целиком из одного и того же окна, поэтому повтор безопасен.
    Возвращает пути к частям, месяц батча и подписи месяцев train и test.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train, valid, test = split_by_time(pd.read_parquet(clean_info["clean_path"]))
    result = {}
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        path = SPLITS_DIR / f"{name}_raw.parquet"
        part.to_parquet(path, index=False)
        result[f"{name}_raw_path"] = str(path)

    history = pd.concat([train, valid])["transaction_time"].dt.to_period("M")
    train_months = sorted(history.unique().astype(str))
    test_month = str(test["transaction_time"].dt.to_period("M").iloc[0])
    logger.info(
        "Train месяцы %s: train %s строк до %s, valid %s строк, test %s %s строк.",
        train_months,
        len(train),
        train["transaction_time"].max(),
        len(valid),
        test_month,
        len(test),
    )
    return {
        **result,
        "dataset_path": clean_info["dataset_path"],
        "batch_month": clean_info["batch_month"],
        "train_months": ",".join(train_months),
        "test_month": test_month,
    }
