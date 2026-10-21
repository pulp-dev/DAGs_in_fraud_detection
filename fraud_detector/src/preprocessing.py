import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CAT_FEATURES = ["merch", "cat_id", "gender", "one_city", "us_state", "jobs"]
NUM_FEATURES = [
    "amount",
    "amount_cat_ratio",
    "lat",
    "lon",
    "merchant_lat",
    "merchant_lon",
    "haversine",
    "population_city",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "is_night",
]
FEATURES = CAT_FEATURES + NUM_FEATURES
LABEL = "target"
MISSING = "missing"
GENDERS = {"M", "F"}


def haversine(lat1, lon1, lat2, lon2):
    rad = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return rad * c


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит категории к строкам, пропуски к MISSING, неизвестный пол к unknown."""
    df_ = df.copy()
    for column in CAT_FEATURES:
        df_[column] = df_[column].fillna(MISSING).astype(str).str.strip()
    gender = df_["gender"].str.upper()
    df_["gender"] = gender.where(gender.isin(GENDERS), "unknown")
    return df_


def fit_stats(train: pd.DataFrame) -> dict:
    """Считает на train статистики, которые нужны признакам и хранятся вместе с моделью."""
    cats = normalize_categories(train)
    by_cat = cats.groupby("cat_id")["amount"].median()
    return {
        "amount_median_global": float(cats["amount"].median()),
        "amount_median_by_cat": {str(k): float(v) for k, v in by_cat.items()},
    }


def run_preproc(input_df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Строит признаки по каждой строке отдельно, поэтому годится и для батча, и для одного сообщения.

    Абсолютная дата (год, месяц, число) в признаки не попадает: модель учится
    на прошлых месяцах, а скорит поток с текущим временем.
    """
    df_ = normalize_categories(input_df)
    times = pd.to_datetime(df_["transaction_time"])
    hour = times.dt.hour

    medians = df_["cat_id"].map(stats["amount_median_by_cat"])
    medians = medians.fillna(stats["amount_median_global"]).astype(float)

    df_["amount_cat_ratio"] = df_["amount"].astype(float) / medians.where(medians > 0, 1.0)
    df_["haversine"] = haversine(df_["lat"], df_["lon"], df_["merchant_lat"], df_["merchant_lon"])
    df_["hour"] = hour
    df_["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df_["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df_["day_of_week"] = times.dt.dayofweek
    df_["is_night"] = ((hour >= 22) | (hour < 6)).astype(int)

    columns = FEATURES + ([LABEL] if LABEL in df_.columns else [])
    output_df = df_[columns].reset_index(drop=True)
    logger.info('Added new features. Output shape: %s', output_df.shape)
    return output_df
