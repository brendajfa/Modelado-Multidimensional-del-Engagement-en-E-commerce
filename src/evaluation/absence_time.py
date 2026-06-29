import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def calculate_absence_time(df_category: pd.DataFrame) -> float:
    """Devuelve el tiempo medio de ausencia en días entre eventos por usuario."""
    if df_category.empty:
        return np.nan

    df_category = df_category.copy()
    df_category["event_time"] = pd.to_datetime(df_category["event_time"], errors="coerce")
    df_category = df_category.sort_values(["user_id", "event_time"])
    df_category["absence_time"] = df_category.groupby("user_id")["event_time"].diff().dt.days

    return df_category["absence_time"].dropna().mean()


def compute_absence_time_summary(
    workpath: str,
    category_codes: Iterable[str],
    source_folder: str = "categories_2019Nov",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Calcula el tiempo medio de ausencia para cada category_code
    leyendo parquet individuales desde workpath/categories/source_folder.
    """
    results = []

    for category_code in tqdm(category_codes, disable=not show_progress):
        input_path = os.path.join(
            workpath,
            "categories",
            source_folder,
            f"{category_code}.parquet",
        )
        try:
            df = pd.read_parquet(input_path)
            absence_time_mean = calculate_absence_time(df)
            results.append(
                {
                    "category_code": category_code,
                    "absence_time_mean": absence_time_mean,
                }
            )
        except Exception as exc:
            print(f"Error processing category {category_code}: {exc}")

    summary_df = pd.DataFrame(results)
    return summary_df