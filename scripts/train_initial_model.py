"""Обучает стартовую модель initial на окне 2019-01..2019-03 тем же кодом, что и DAG.

Результат: fraud_detector/models/my_catboost.cbm и my_catboost.stats.json,
которые скоринг копирует в общий каталог, пока DAG не опубликует свою версию.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FRAUD_SRC", str(ROOT / "fraud_detector" / "src"))
sys.path.insert(0, str(ROOT))

from pipeline.retrain.clean_dataset import clean_frame  # noqa: E402
from pipeline.retrain.fit_catboost import evaluate, train_model  # noqa: E402
from pipeline.retrain.preprocess_dataset import load_preprocessing  # noqa: E402
from pipeline.retrain.time_train_test_split import split_by_time  # noqa: E402

WINDOW_START = "2019-01-01"
WINDOW_END_EXCLUSIVE = "2019-04-01"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "fraud_detector" / "models"))
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])
    times = frame["transaction_time"]
    frame = frame.loc[(times >= WINDOW_START) & (times < WINDOW_END_EXCLUSIVE)]

    cleaned, dropped = clean_frame(frame)
    train, valid, test = split_by_time(cleaned)
    preprocessing = load_preprocessing()
    stats = preprocessing.fit_stats(train)
    parts = [preprocessing.run_preproc(part, stats) for part in (train, valid, test)]
    cat_features = preprocessing.CAT_FEATURES
    model = train_model(parts[0], parts[1], cat_features)
    metrics = evaluate(model, parts[2], cat_features)

    out_dir = Path(args.out_dir)
    model.save_model(str(out_dir / "my_catboost.cbm"))
    (out_dir / "my_catboost.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(
        {
            "dropped": dropped,
            "rows": {"train": len(train), "valid": len(valid), "test": len(test)},
            "best_iteration": model.get_best_iteration(),
            "test_metrics": metrics,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
