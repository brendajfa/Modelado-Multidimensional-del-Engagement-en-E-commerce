import os
from typing import Tuple

import pandas as pd
from pathlib import Path

import numpy as np
from src.config import WORKPATH as workpath
from src.config import RESULTSPATH as resultspath


analysis_dir = os.path.join(resultspath, "validation_analysis")

def _load_month(month_name: str) -> pd.DataFrame:
    path = os.path.join(workpath, "raw", f"2019-{month_name}-cleaned.parquet")
    df = pd.read_parquet(path)
    df["temporalidad"] = month_name.upper()[:3]
    return df.drop_duplicates(["event_type", "user_session", "product_id", "price"])


def compute_cluster_kpi_reports(
    df_nov: pd.DataFrame,
    df_dec: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # df_dec = _load_month("Dec")
    # df_nov = _load_month("Nov")

    clusters = pd.read_parquet(os.path.join(resultspath, "clusters", "training_clusters.parquet"))
    df_entero = pd.concat([df_nov, df_dec], ignore_index=True)

    def _aggregate(cluster_col: str) -> pd.DataFrame:
        return (
            df_entero.merge(clusters, on="category_code")
            .groupby(["temporalidad", cluster_col])
            .agg(
                precio_prom=("price", "mean"),
                compras=("event_type", lambda x: (x == "purchase").sum()),
                conversion_prom=("conversion", "mean"),
            )
            .reset_index()
        )

    cluster_generic = _aggregate("cluster_generic")
    cluster_user_based = _aggregate("cluster_user_based")
    cluster_time_based = _aggregate("cluster_time_based")

    os.makedirs(analysis_dir, exist_ok=True)

    cluster_generic.to_parquet(os.path.join(analysis_dir, "cluster_generic.parquet"))
    cluster_user_based.to_parquet(os.path.join(analysis_dir, "cluster_user_based.parquet"))
    cluster_time_based.to_parquet(os.path.join(analysis_dir, "cluster_time_based.parquet"))

    return cluster_generic, cluster_user_based, cluster_time_based


def compute_category_elasticity_reports(
    df_nov: pd.DataFrame,
    df_dec: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_entero = pd.concat([df_nov, df_dec], ignore_index=True)

    compras = (
        df_entero
        .groupby(["temporalidad", "category_code"])
        .agg(
            precio_prom=("price", "mean"),
            compras=("event_type", lambda x: (x == "purchase").sum()),
        )
        .reset_index()
    )

    compras_pivot = compras.pivot(index="category_code", columns="temporalidad")
    compras_pivot = compras_pivot.sort_index(axis=1, level=0)

    compras_dec = compras_pivot[("compras", "DEC")].replace(0, np.nan)
    compras_nov = compras_pivot[("compras", "NOV")].replace(0, np.nan)
    precio_dec = compras_pivot[("precio_prom", "DEC")].replace(0, np.nan)
    precio_nov = compras_pivot[("precio_prom", "NOV")].replace(0, np.nan)

    compras_pivot[("elasticidad", "")] = (
        (compras_dec - compras_nov) / compras_nov
    ) / (
        (precio_dec - precio_nov) / precio_nov
    )
    compras_pivot[("elasticidad", "")] = compras_pivot[("elasticidad", "")].replace([np.inf, -np.inf], np.nan)

    eventos = (
        df_entero
        .groupby(["temporalidad", "category_code", "event_type"])
        .size()
        .reset_index(name="conteo")
    )

    conversion_mes = (
        df_entero
        .groupby(["temporalidad", "category_code"])["conversion"]
        .mean()
        .reset_index()
    )

    analysis_dir.mkdir(parents=True, exist_ok=True)

    compras_pivot.to_parquet(analysis_dir / "compras.parquet")
    eventos.to_parquet(analysis_dir / "eventos.parquet")
    conversion_mes.to_parquet(analysis_dir / "conversion_mes.parquet")

    return compras_pivot, eventos, conversion_mes


def _normalize_pivot_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df.columns = [
        f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] != "" else str(col)
        for col in df.columns
    ]
    return df


def load_analysis_artifacts() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    compras = pd.read_parquet(os.path.join(analysis_dir, "compras.parquet"))
    eventos = pd.read_parquet(os.path.join(analysis_dir, "eventos.parquet"))
    conversion_mes = pd.read_parquet(os.path.join(analysis_dir, "conversion_mes.parquet"))
    clusters = pd.read_parquet(os.path.join(resultspath, "clusters", "training_clusters.parquet"))
    return compras, eventos, conversion_mes, clusters


def compute_category_elasticity_summary(
    compras: pd.DataFrame,
    clusters: pd.DataFrame,
    min_compras: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # compras, eventos, conversion_mes, clusters = load_analysis_artifacts()

    compras_reset = _normalize_pivot_columns(compras)
    df_merged = compras_reset.merge(clusters, on="category_code", how="left")

    df_merged_not_inf = df_merged.replace([np.inf, -np.inf], np.nan).dropna()

    df_merged_not_inf["elastico_no_elastico"] = (
        df_merged_not_inf["elasticidad"] > 1
    ).astype(int)

    df_filtered = df_merged_not_inf[
        (df_merged_not_inf["compras_NOV"] > min_compras)
        & (df_merged_not_inf["compras_DEC"] > min_compras)
    ]

    summary = (
        df_filtered
        .groupby(["cluster_generic", "elastico_no_elastico"])["elasticidad"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )

    return df_filtered, summary