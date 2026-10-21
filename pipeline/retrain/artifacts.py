"""Сохранение черновиков модели под постоянными именами."""

import json
import os
from pathlib import Path

from pipeline.retrain.quality_gate import decide

SKIP_KEYS = {"staging_model", "staging_stats", "dataset_path", "test_raw_path"}


def store_artifacts(info: dict, target_dir: Path, decision: str) -> Path:
    """Переносит черновики .cbm и .stats.json в target_dir и пишет metrics.json.

    Если черновика уже нет, а итоговый файл есть, шаг пропускается,
    поэтому повтор после сбоя не падает и не плодит копии.
    Возвращает путь к итоговому .cbm.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"my_catboost_{info['version']}"
    final_model = target_dir / f"{prefix}.cbm"
    for staging, final in (
        (Path(info["staging_model"]), final_model),
        (Path(info["staging_stats"]), target_dir / f"{prefix}.stats.json"),
    ):
        if staging.exists():
            os.replace(staging, final)
        elif not final.exists():
            raise FileNotFoundError(f"Нет артефакта модели: {staging}")

    _, reason = decide(info)
    metrics = {key: value for key, value in info.items() if key not in SKIP_KEYS}
    metrics.update(decision=decision, reason=reason)
    (target_dir / f"{prefix}.metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return final_model
