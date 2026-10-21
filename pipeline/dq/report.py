"""HTML-отчет Data Quality."""

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from pipeline.db import get_conn
from pipeline.settings import DQ_REPORTS_DIR, TRAINING_DIR

logger = logging.getLogger(__name__)

STAGING_DIR = TRAINING_DIR / "dq"

TEMPLATE = Template("""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DQ отчет {{ run_id }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; color: #1f2328; background: #fff; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .meta { color: #59636e; margin-bottom: 20px; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { border: 1px solid #d1d9e0; border-radius: 8px; padding: 10px 16px; min-width: 110px; }
  .card b { display: block; font-size: 22px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border-bottom: 1px solid #d1d9e0; padding: 8px; text-align: left; vertical-align: top; }
  th { background: #f6f8fa; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .status { font-weight: 600; padding: 2px 8px; border-radius: 10px; }
  .pass { background: #dafbe1; color: #1a7f37; }
  .warn { background: #fff8c5; color: #9a6700; }
  .fail { background: #ffebe9; color: #cf222e; }
</style>
</head>
<body>
<h1>Data Quality: labeled_transactions</h1>
<div class="meta">Прогон {{ run_id }}, сформирован {{ generated_at }} UTC.
Строк в витрине: {{ total_rows }}, период {{ first_time }} - {{ last_time }}.</div>
<div class="cards">
  <div class="card">Итог<b class="status {{ overall }}">{{ overall | upper }}</b></div>
  <div class="card">Правил<b>{{ results | length }}</b></div>
  <div class="card">pass<b>{{ counts.pass }}</b></div>
  <div class="card">warn<b>{{ counts.warn }}</b></div>
  <div class="card">fail<b>{{ counts.fail }}</b></div>
</div>
<table>
  <tr><th>Правило</th><th>Группа</th><th>Описание</th><th>Важность</th>
      <th>Проверено</th><th>Нарушений</th><th>Доля</th><th>Порог</th><th>Статус</th></tr>
  {% for r in results %}
  <tr>
    <td>{{ r.name }}</td><td>{{ r.group }}</td><td>{{ r.description }}</td><td>{{ r.severity }}</td>
    <td class="num">{{ r.checked }}</td><td class="num">{{ r.failed }}</td>
    <td class="num">{{ "%.4f%%" | format(r.share * 100) }}</td>
    <td class="num">{{ "%.4f%%" | format(r.max_share * 100) }}</td>
    <td><span class="status {{ r.status }}">{{ r.status | upper }}</span></td>
  </tr>
  {% endfor %}
</table>
</body>
</html>
""")


def render_report(completeness: list[dict], validity: list[dict], consistency: list[dict], ts_nodash=None) -> str:
    """Собирает результаты всех групп в HTML.

    Читает результаты проверок из XCom и сводку витрины из labeled_transactions.
    Пишет черновик training_data/dq/dq_report_<ts>.html, где ts это логическая дата прогона,
    поэтому повтор перезаписывает тот же файл, а не создает новый.
    Возвращает путь к черновику.
    """
    results = completeness + validity + consistency
    counts = {status: sum(r["status"] == status for r in results) for status in ("pass", "warn", "fail")}
    overall = "fail" if counts["fail"] else ("warn" if counts["warn"] else "pass")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), min(transaction_time), max(transaction_time) FROM labeled_transactions")
            total_rows, first_time, last_time = cur.fetchone()
    finally:
        conn.close()

    html = TEMPLATE.render(
        run_id=ts_nodash,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        total_rows=total_rows,
        first_time=first_time,
        last_time=last_time,
        results=results,
        counts=counts,
        overall=overall,
    )
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGING_DIR / f"dq_report_{ts_nodash}.html"
    path.write_text(html, encoding="utf-8")
    logger.info("DQ итог %s: %s", overall, counts)
    return str(path)


def publish_report(staged_path: str) -> str:
    """Публикует отчет в каталог DQ_REPORTS_DIR рядом с каталогом моделей.

    Копирует черновик в dq_report_<ts>.html и latest.html через временный файл
    и os.replace, поэтому читатель не увидит недописанный отчет,
    а повтор просто перезаписывает те же файлы.
    Возвращает путь к опубликованному отчету.
    """
    staged = Path(staged_path)
    DQ_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in (staged.name, "latest.html"):
        tmp = DQ_REPORTS_DIR / f".{name}.tmp"
        shutil.copyfile(staged, tmp)
        os.replace(tmp, DQ_REPORTS_DIR / name)
    logger.info("DQ отчет опубликован: %s", DQ_REPORTS_DIR / staged.name)
    return str(DQ_REPORTS_DIR / staged.name)
