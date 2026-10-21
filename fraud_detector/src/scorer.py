import json
import logging
import os
import shutil
import threading
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from preprocessing import run_preproc


logger = logging.getLogger(__name__)

THRESHOLD = 0.5
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))
BUNDLED_MODEL = Path(os.getenv("BUNDLED_MODEL_PATH", "/app/models/my_catboost.cbm"))
BUNDLED_STATS = BUNDLED_MODEL.with_suffix(".stats.json")

_lock = threading.Lock()
_model = None
_stats = None
_version = None


def _model_path(version: str) -> Path:
    return MODELS_DIR / f"my_catboost_{version}.cbm"


def _stats_path(version: str) -> Path:
    return MODELS_DIR / f"my_catboost_{version}.stats.json"


def ensure_bootstrap():
    """Кладет зашитую модель и ее статистики в общий каталог, пока DAG не опубликует новую."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    latest = MODELS_DIR / "latest.txt"
    if latest.exists() and latest.read_text(encoding="utf-8").strip():
        return
    for source, dest in ((BUNDLED_MODEL, _model_path("initial")), (BUNDLED_STATS, _stats_path("initial"))):
        if dest.exists():
            continue
        if not source.exists():
            raise FileNotFoundError(f"Нет исходного артефакта: {source}")
        shutil.copyfile(source, dest)
        logger.info("Скопирован исходный артефакт в %s", dest)
    latest.write_text("initial\n", encoding="utf-8")


def _read_latest_version() -> str:
    ensure_bootstrap()
    version = (MODELS_DIR / "latest.txt").read_text(encoding="utf-8").strip()
    if not version:
        raise RuntimeError(f"Пустой указатель модели: {MODELS_DIR / 'latest.txt'}")
    return version


def _load_artifacts(version: str) -> tuple[CatBoostClassifier, dict]:
    model_path = _model_path(version)
    stats_path = _stats_path(version)
    for path in (model_path, stats_path):
        if not path.exists():
            raise FileNotFoundError(f"Артефакт версии {version} не найден: {path}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    logger.info("Загружена модель версии %s из %s", version, model_path)
    return model, stats


def _current_artifacts():
    """Возвращает модель и статистики, которые соответствуют latest.txt. Вызывать под _lock."""
    global _model, _stats, _version
    version = _read_latest_version()
    if _model is None or version != _version:
        _model, _stats = _load_artifacts(version)
        _version = version
    return _model, _stats, _version


def make_pred(raw_df, source_info="kafka"):
    """Строит признаки статистиками текущей версии и возвращает скор, флаг и версию модели."""
    with _lock:
        model, stats, model_version = _current_artifacts()
        features = run_preproc(raw_df, stats)
        names = list(model.feature_names_)
        missing = [column for column in names if column not in features.columns]
        if missing:
            raise ValueError(f"Во входных данных нет признаков модели: {missing}")
        y_proba = model.predict_proba(features[names])[:, 1]

    submission = pd.DataFrame({
        "score": y_proba,
        "fraud_flag": (y_proba > THRESHOLD) * 1,
    })
    logger.info("Prediction complete for data from %s, model_version=%s", source_info, model_version)
    return submission, y_proba, model_version
