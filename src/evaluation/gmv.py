import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm.auto import tqdm


def _calculate_avg_price(df: pd.DataFrame) -> float:
    return df["price"].mean() if not df.empty else 0.0


def _calculate_view_to_cart_rate(df: pd.DataFrame) -> float:
    view_events = (df["event_type"] == "view").sum()
    cart_events = (df["event_type"] == "cart").sum()
    return cart_events / view_events if view_events > 0 else 0.0


def _calculate_cart_to_purchase_rate(df: pd.DataFrame) -> float:
    cart_events = (df["event_type"] == "cart").sum()
    purchase_events = (df["event_type"] == "purchase").sum()
    return purchase_events / cart_events if cart_events > 0 else 0.0


def _calculate_total_purchase_probability(df: pd.DataFrame) -> float:
    total_events = len(df)
    purchase_events = (df["event_type"] == "purchase").sum()
    return purchase_events / total_events if total_events > 0 else 0.0


def _calculate_expected_gmv(df: pd.DataFrame) -> float:
    purchase_df = df[df["event_type"] == "purchase"]
    return purchase_df["price"].sum() if not purchase_df.empty else 0.0


def compute_category_gmv_summary(
    workpath: str,
    source_folder: str,
    category_codes: Iterable[str],
    output_path: Optional[str] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Calcula métricas de GMV por category_code a partir de parquet individuales.

    Parámetros
    ----------
    workpath : str
        Ruta base del proyecto.
    source_folder : str
        Carpeta bajo workpath/categories donde están los ficheros, p.ej.
        "training_201911" o "validation_201912".
    category_codes : Iterable[str]
        Lista de códigos de categoría a procesar.
    output_path : Optional[str]
        Si se proporciona, guarda el resultado en parquet en esta ruta.
    show_progress : bool
        Mostrar barra de progreso.
    """
    rows = []

    for category_code in tqdm(category_codes, disable=not show_progress):
        input_path = os.path.join(
            workpath,
            "categories",
            source_folder,
            f"{category_code}.parquet",
        )
        try:
            df = pd.read_parquet(input_path)

            avg_price = _calculate_avg_price(df)
            view_to_cart_rate = _calculate_view_to_cart_rate(df)
            cart_to_purchase_rate = _calculate_cart_to_purchase_rate(df)
            total_purchase_probability = _calculate_total_purchase_probability(df)
            expected_gmv = _calculate_expected_gmv(df)

            rows.append({
                "category_code": category_code,
                "average_price": avg_price,
                "view_to_cart_rate": view_to_cart_rate,
                "cart_to_purchase_rate": cart_to_purchase_rate,
                "total_purchase_probability": total_purchase_probability,
                "expected_gmv": expected_gmv,
            })

        except Exception as exc:
            print(f"Error processing category {category_code}: {exc}")

    summary_df = pd.DataFrame(rows)

    for col in [
        "view_to_cart_rate",
        "cart_to_purchase_rate",
        "total_purchase_probability",
    ]:
        summary_df[col] = (summary_df[col] * 100).round(2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_parquet(output_path)

    return summary_df
