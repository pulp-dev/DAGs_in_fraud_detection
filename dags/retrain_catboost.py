"""DAG скользящего переобучения CatBoost.

Файл только связывает задачи. Логика каждой задачи лежит в pipeline.retrain.
"""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.retrain.advance_window import advance_window
from pipeline.retrain.clean_dataset import clean_dataset
from pipeline.retrain.evaluate_champion import evaluate_champion
from pipeline.retrain.extract_month_batch import extract_month_batch
from pipeline.retrain.fit_catboost import fit_catboost
from pipeline.retrain.preprocess_dataset import preprocess_dataset
from pipeline.retrain.publish_model import publish_model
from pipeline.retrain.quality_gate import quality_gate
from pipeline.retrain.reject_model import reject_model
from pipeline.retrain.slide_training_window import slide_training_window
from pipeline.retrain.time_train_test_split import time_train_test_split


@dag(
    dag_id="retrain_catboost",
    description="Раз в 5 минут сдвигает 3-месячное окно и переобучает CatBoost",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=15),
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["mlops", "catboost"],
    doc_md="""
### Переобучение my_catboost

Каждый запуск имитирует новый месяц в проде:

1. Из витрины `labeled_transactions` берется следующий календарный месяц.
2. Окно из трех месяцев, последний из которых новый, собирается заново из витрины.
3. Очистка: пропуски, невалидные target, amount и координаты, дубликаты.
4. Test - самый свежий месяц. Ранние месяцы делятся по времени на train и valid,
   valid нужен для ранней остановки.
5. Препроцессинг из `fraud_detector/src`: статистики считаются на train
   и публикуются вместе с моделью.
6. CatBoost обучается с ранней остановкой по valid.
7. Prod-модель из `latest.txt` оценивается на том же test-месяце.
8. Quality gate: если PR-AUC новой модели ниже prod, она уходит в `models/rejected`,
   иначе публикуется и `latest.txt` переключается на нее.

Указатель месяца в `training_state` сдвигается только после сохранения модели
в одной из веток gate.
Если очередного месяца в витрине еще нет, задачи помечаются как skipped.
Новые месяцы после исторических данных дописывает DAG `supply_labeled_batch`.
    """,
)
def retrain_catboost():
    """Связывает подготовку окна, обучение, quality gate, публикацию или отказ и сдвиг окна."""
    extracted = task(extract_month_batch, task_id="extract_month_batch")()
    slid = task(slide_training_window, task_id="slide_training_window")(extracted)
    cleaned = task(clean_dataset, task_id="clean_dataset")(slid)
    split = task(time_train_test_split, task_id="time_train_test_split")(cleaned)
    processed = task(preprocess_dataset, task_id="preprocess_dataset")(split)
    trained = task(fit_catboost, task_id="fit_catboost")(processed)
    evaluated = task(evaluate_champion, task_id="evaluate_champion")(trained)
    gate = task.branch(quality_gate, task_id="quality_gate")(evaluated)
    published = task(publish_model, task_id="publish_model")(evaluated)
    rejected = task(reject_model, task_id="reject_model")(evaluated)
    advanced = task(
        advance_window,
        task_id="advance_window",
        trigger_rule="none_failed_min_one_success",
    )(evaluated)
    gate >> [published, rejected] >> advanced


retrain_catboost()
