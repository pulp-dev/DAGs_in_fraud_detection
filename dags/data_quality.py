"""DAG проверки качества размеченной витрины.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.dq.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.dq.checks import check_completeness, check_consistency, check_validity
from pipeline.dq.report import publish_report, render_report


@dag(
    dag_id="data_quality",
    description="Раз в 10 минут проверяет витрину labeled_transactions и публикует HTML-отчет",
    schedule="*/10 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "data-quality"],
    doc_md="""
### Data Quality витрины

Три группы правил считаются SQL-агрегатами в Postgres:
completeness, validity и consistency. Результаты собираются в HTML-отчет,
который публикуется в `dq_reports` рядом с каталогом моделей
(`dq_report_<ts>.html` и `latest.html`).
Нарушения правил не роняют DAG: статус каждого правила виден в отчете.
    """,
)
def data_quality():
    """Связывает три группы проверок, сборку отчета и его публикацию."""
    completeness = task(check_completeness, task_id="check_completeness")()
    validity = task(check_validity, task_id="check_validity")()
    consistency = task(check_consistency, task_id="check_consistency")()
    staged = task(render_report, task_id="render_report")(completeness, validity, consistency)
    task(publish_report, task_id="publish_report")(staged)


data_quality()
