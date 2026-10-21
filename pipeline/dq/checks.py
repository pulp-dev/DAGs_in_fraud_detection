"""Выполнение правил Data Quality по группам."""

import logging

from pipeline.db import get_conn
from pipeline.dq.rules import RULES

logger = logging.getLogger(__name__)


def run_rules(group: str) -> list[dict]:
    """Выполняет правила группы и возвращает результат по каждому.

    Статус pass, если доля нарушений не выше max_share, иначе fail для critical
    и warn для warning.
    """
    results = []
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for rule in (rule for rule in RULES if rule["group"] == group):
                cur.execute(rule["sql"])
                checked, failed = cur.fetchone()
                share = failed / checked if checked else 0.0
                passed = share <= rule["max_share"]
                status = "pass" if passed else ("fail" if rule["severity"] == "critical" else "warn")
                results.append({
                    "name": rule["name"],
                    "group": group,
                    "severity": rule["severity"],
                    "description": rule["description"],
                    "checked": int(checked),
                    "failed": int(failed),
                    "share": share,
                    "max_share": rule["max_share"],
                    "status": status,
                })
                logger.info("%s: %s, нарушений %s из %s", rule["name"], status, failed, checked)
    finally:
        conn.close()
    return results


def check_completeness() -> list[dict]:
    """Проверяет полноту обязательных колонок витрины.

    Только читает labeled_transactions и ничего не пишет, поэтому повтор безопасен.
    Возвращает результаты правил группы completeness.
    """
    return run_rules("completeness")


def check_validity() -> list[dict]:
    """Проверяет допустимые значения target, amount, координат и gender.

    Только читает labeled_transactions и ничего не пишет, поэтому повтор безопасен.
    Возвращает результаты правил группы validity.
    """
    return run_rules("validity")


def check_consistency() -> list[dict]:
    """Проверяет дубликаты, объем и долю фрода по месяцам и глубину истории.

    Только читает labeled_transactions и ничего не пишет, поэтому повтор безопасен.
    Возвращает результаты правил группы consistency.
    """
    return run_rules("consistency")
