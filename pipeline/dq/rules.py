"""Правила Data Quality для витрины labeled_transactions.

Каждое правило это SQL, который возвращает (checked, failed): сколько объектов
проверено и сколько нарушают правило. Правило выполнено, если
failed / checked <= max_share.
"""

from pipeline.settings import HISTORY_YEARS

MIN_MONTH_ROWS = 10_000
MAX_AMOUNT = 30_000
FRAUD_RATE_RANGE = (0.001, 0.03)

MONTHS_CTE = """
    WITH bounds AS (
        SELECT date_trunc('month', min(transaction_time)) AS first_month,
               date_trunc('month', max(transaction_time)) AS last_month
        FROM labeled_transactions
    ),
    months AS (
        SELECT generate_series(first_month, last_month, interval '1 month') AS month FROM bounds
    ),
    stats AS (
        SELECT date_trunc('month', transaction_time) AS month,
               count(*) AS row_count,
               avg(target) AS fraud_rate
        FROM labeled_transactions
        GROUP BY 1
    )
"""

RULES = [
    {
        "name": "completeness",
        "group": "completeness",
        "severity": "critical",
        "description": "Нет NULL в transaction_time, amount, merch, cat_id, координатах и target",
        "max_share": 0.0,
        "sql": """
            SELECT count(*),
                   count(*) FILTER (WHERE transaction_time IS NULL OR amount IS NULL
                       OR merch IS NULL OR cat_id IS NULL OR lat IS NULL OR lon IS NULL
                       OR merchant_lat IS NULL OR merchant_lon IS NULL OR target IS NULL)
            FROM labeled_transactions
        """,
    },
    {
        "name": "target_domain",
        "group": "validity",
        "severity": "critical",
        "description": "target принимает только значения 0 и 1",
        "max_share": 0.0,
        "sql": """
            SELECT count(*), count(*) FILTER (WHERE target NOT IN (0, 1))
            FROM labeled_transactions
        """,
    },
    {
        "name": "amount_range",
        "group": "validity",
        "severity": "critical",
        "description": f"Сумма транзакции в диапазоне (0, {MAX_AMOUNT}]",
        "max_share": 0.0,
        "sql": f"""
            SELECT count(*), count(*) FILTER (WHERE amount <= 0 OR amount > {MAX_AMOUNT})
            FROM labeled_transactions
        """,
    },
    {
        "name": "coordinates_range",
        "group": "validity",
        "severity": "critical",
        "description": "Широта клиента и мерчанта в [-90, 90], долгота в [-180, 180]",
        "max_share": 0.0,
        "sql": """
            SELECT count(*),
                   count(*) FILTER (WHERE lat NOT BETWEEN -90 AND 90
                       OR merchant_lat NOT BETWEEN -90 AND 90
                       OR lon NOT BETWEEN -180 AND 180
                       OR merchant_lon NOT BETWEEN -180 AND 180)
            FROM labeled_transactions
        """,
    },
    {
        "name": "gender_domain",
        "group": "validity",
        "severity": "warning",
        "description": "gender принимает только значения M и F",
        "max_share": 0.0,
        "sql": """
            SELECT count(*), count(*) FILTER (WHERE gender IS NULL OR gender NOT IN ('M', 'F'))
            FROM labeled_transactions
        """,
    },
    {
        "name": "duplicates",
        "group": "consistency",
        "severity": "warning",
        "description": "Доля полных дубликатов строк не больше 0.1%",
        "max_share": 0.001,
        "sql": """
            SELECT count(*), count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM labeled_transactions) d)
            FROM labeled_transactions
        """,
    },
    {
        "name": "monthly_volume",
        "group": "consistency",
        "severity": "critical",
        "description": f"Нет пропущенных месяцев, в каждом месяце не меньше {MIN_MONTH_ROWS} строк",
        "max_share": 0.0,
        "sql": MONTHS_CTE + f"""
            SELECT count(*), count(*) FILTER (WHERE coalesce(stats.row_count, 0) < {MIN_MONTH_ROWS})
            FROM months LEFT JOIN stats USING (month)
        """,
    },
    {
        "name": "fraud_rate_per_month",
        "group": "consistency",
        "severity": "warning",
        "description": (
            f"Доля фрода каждого месяца в [{FRAUD_RATE_RANGE[0]:.1%}, {FRAUD_RATE_RANGE[1]:.1%}]"
        ),
        "max_share": 0.0,
        "sql": MONTHS_CTE + f"""
            SELECT count(*), count(*) FILTER (
                WHERE fraud_rate NOT BETWEEN {FRAUD_RATE_RANGE[0]} AND {FRAUD_RATE_RANGE[1]})
            FROM stats
        """,
    },
    {
        "name": "history_depth",
        "group": "consistency",
        "severity": "warning",
        "description": f"Нет наблюдений старше {HISTORY_YEARS} лет от самого свежего месяца",
        "max_share": 0.0,
        "sql": f"""
            SELECT count(*),
                   count(*) FILTER (WHERE transaction_time < (
                       SELECT date_trunc('month', max(transaction_time)) - interval '{HISTORY_YEARS} years'
                       FROM labeled_transactions))
            FROM labeled_transactions
        """,
    },
]
