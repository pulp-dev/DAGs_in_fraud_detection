"""Решение, публиковать ли новую модель."""

import logging

from pipeline.settings import GATE_METRIC, GATE_TOLERANCE

logger = logging.getLogger(__name__)


def decide(gate_info: dict) -> tuple[bool, str]:
    """Сравнивает GATE_METRIC новой и prod-модели на одном test-месяце.

    Новая модель проходит, если ее метрика не ниже prod минус GATE_TOLERANCE.
    Без prod-модели проходит. Если метрика новой модели не определена, не проходит.
    """
    new = gate_info.get(GATE_METRIC)
    champion = gate_info.get("champion_metrics") or {}
    old = champion.get(GATE_METRIC)
    if new is None:
        return False, f"{GATE_METRIC} новой модели не определен"
    if old is None:
        return True, f"нет метрики prod-модели, {GATE_METRIC} новой {new:.4f}"
    verdict = ">=" if new >= old - GATE_TOLERANCE else "<"
    reason = f"{GATE_METRIC} новой {new:.4f} {verdict} prod {old:.4f} - допуск {GATE_TOLERANCE}"
    return verdict == ">=", reason


def quality_gate(gate_info: dict) -> str:
    """Выбирает ветку DAG: publish_model или reject_model.

    Читает только метрики из gate_info и ничего не пишет, поэтому повтор
    дает то же решение.
    """
    approved, reason = decide(gate_info)
    logger.info("Quality gate %s: %s", "пройден" if approved else "не пройден", reason)
    return "publish_model" if approved else "reject_model"
