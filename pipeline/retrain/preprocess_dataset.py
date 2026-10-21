"""Препроцессинг train, valid и test тем же кодом, что и онлайн-скоринг."""

import json
import logging
import sys

import pandas as pd

from pipeline.settings import FRAUD_SRC, SPLITS_DIR

logger = logging.getLogger(__name__)

PARTS = ("train", "valid", "test")


def load_preprocessing():
    """Импортирует preprocessing из fraud_detector/src, общий для DAG и скоринга."""
    if FRAUD_SRC not in sys.path:
        sys.path.insert(0, FRAUD_SRC)
    import preprocessing

    return preprocessing


def preprocess_dataset(split_info: dict) -> dict:
    """Считает статистики признаков на train и строит признаки для train, valid и test.

    Читает splits/<part>_raw.parquet по путям из split_info. Статистики считаются
    только по train, чтобы valid и test не протекали в признаки.
    Пишет splits/<part>_proc.parquet и splits/stats.json.
    Входные файлы не меняются, выходные перезаписываются целиком, поэтому повтор безопасен.
    Возвращает пути к частям и статистикам, список категориальных признаков и метаданные месяца.
    """
    preprocessing = load_preprocessing()
    raw = {part: pd.read_parquet(split_info[f"{part}_raw_path"]) for part in PARTS}
    stats = preprocessing.fit_stats(raw["train"])
    stats_path = SPLITS_DIR / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {}
    for part, frame in raw.items():
        processed = preprocessing.run_preproc(frame, stats)
        path = SPLITS_DIR / f"{part}_proc.parquet"
        processed.to_parquet(path, index=False)
        result[f"{part}_proc_path"] = str(path)
        logger.info("После препроцессинга %s: %s", part, processed.shape)
    return {
        **result,
        "stats_path": str(stats_path),
        "test_raw_path": split_info["test_raw_path"],
        "cat_features": preprocessing.CAT_FEATURES,
        "dataset_path": split_info["dataset_path"],
        "batch_month": split_info["batch_month"],
        "train_months": split_info["train_months"],
        "test_month": split_info["test_month"],
    }
