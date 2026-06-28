import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
from typing import Union
import seaborn as sns
from src.utils.style import apply_style
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.patches import Patch


COLORS = apply_style()
BG = COLORS.get("bg", "#FAFAF9")


def plot_ecommerce_weekly_chart(
    df: pd.DataFrame,
    output_path: Union[str, os.PathLike] = "ecommerce_weekly_chart.png",
    show: bool = True,
):
    """
    Genera un gráfico semanal de eventos de e-commerce y ratios de conversión.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener las columnas: event_time, user_session, event_type.
    output_path : str
        Ruta donde guardar la imagen.
    show : bool
        Si True, muestra el gráfico.
    """
    if "event_time" not in df.columns:
        raise ValueError("El DataFrame debe tener la columna 'event_time'")
    if "user_session" not in df.columns:
        raise ValueError("El DataFrame debe tener la columna 'user_session'")
    if "event_type" not in df.columns:
        raise ValueError("El DataFrame debe tener la columna 'event_type'")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])

    # Crear columna semana
    df["week"] = df["event_time"].dt.to_period("W").dt.start_time

    # Contar sesiones por semana y tipo de evento
    weekly = (
        df.groupby(["week", "user_session", "event_type"])
          .size()
          .gt(0)
          .astype(int)
          .reset_index(name="has_event")
          .groupby(["week", "event_type"])["has_event"]
          .sum()
          .unstack(fill_value=0)
          .reindex(columns=["view", "cart", "purchase"], fill_value=0)
          .reset_index()
    )

    weekly = weekly.sort_values("week").reset_index(drop=True)
    weekly["week_label"] = weekly["week"].dt.strftime("%d %b")

    # Ratios
    weekly["ratio_view_to_cart"] = weekly["view"] / weekly["cart"].replace(0, np.nan)
    weekly["ratio_cart_to_purchase"] = weekly["cart"] / weekly["purchase"].replace(0, np.nan)

    # --- Layout ---
    x = np.arange(len(weekly))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#FAFAF9")
    ax1.set_facecolor("#FAFAF9")

    # --- Bars ---
    bars_view = ax1.bar(
        x - width, weekly["view"], width,
        label="Views", color="#378ADD", alpha=0.85
    )
    bars_cart = ax1.bar(
        x, weekly["cart"], width,
        label="Cart", color="#1D9E75", alpha=0.85
    )
    bars_purch = ax1.bar(
        x + width, weekly["purchase"], width,
        label="Purchase", color="#D85A30", alpha=0.85
    )

    ax1.set_ylabel("Eventos", fontsize=11, color="#5F5E5A")
    ax1.tick_params(axis="y", labelcolor="#5F5E5A")
    ax1.set_xticks(x)
    ax1.set_xticklabels(weekly["week_label"], fontsize=10, color="#5F5E5A")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K" if v >= 1e3 else str(int(v))
        )
    )
    ax1.grid(axis="y", color="#D3D1C7", linewidth=0.6, linestyle="--")
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # --- Lines (secondary axis) ---
    ax2 = ax1.twinx()
    line1, = ax2.plot(
        x, weekly["ratio_view_to_cart"],
        color="#7F77DD", linewidth=2.2,
        linestyle="--", marker="o", markersize=6,
        label="Ratio view→cart"
    )
    line2, = ax2.plot(
        x, weekly["ratio_cart_to_purchase"],
        color="#D4537E", linewidth=2.2,
        linestyle=":", marker="D", markersize=6,
        label="Ratio cart→purchase"
    )

    ax2.set_ylabel("Ratio (%)", fontsize=11, color="#5F5E5A")
    ax2.tick_params(axis="y", labelcolor="#5F5E5A")
    for spine in ["top", "left"]:
        ax2.spines[spine].set_visible(False)

    # --- Legend ---
    bar_handles = [bars_view, bars_cart, bars_purch]
    line_handles = [line1, line2]
    all_handles = bar_handles + line_handles
    all_labels = [h.get_label() for h in bar_handles] + [h.get_label() for h in line_handles]

    ax1.legend(
        all_handles, all_labels,
        loc="upper left", fontsize=9,
        framealpha=0.6, edgecolor="#D3D1C7",
        facecolor="#FAFAF9"
    )

    plt.title("Eventos semanales e-commerce — Oct/Nov 2019", fontsize=13, pad=14, color="#2C2C2A")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, weekly


def plot_price_distribution(
    df: pd.DataFrame,
    output_path: Union[str, os.PathLike] | None = None,
    show: bool = True,
):
    """
    Dibuja histograma + KDE y boxplot de la columna 'price'.
    Acepta output_path (str o PathLike) para guardar la figura.
    Devuelve (fig, (ax1, ax2)).
    """
    COLORS = apply_style()

    if "price" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'price'")

    price_series = df["price"].dropna()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histograma + KDE ---
    sns.histplot(
        price_series,
        bins=50,
        kde=True,
        ax=ax1,
        color=COLORS["bars"]["view"],
        alpha=0.85,
    )

    sns.kdeplot(
        price_series,
        ax=ax1,
        color=COLORS["lines"]["ratio_view_cart"],
        linewidth=2.2,
    )

    ax1.set_title("Distribución del precio", color=COLORS["title"])
    ax1.grid(True)

    # --- Boxplot ---
    sns.boxplot(
        data=df,
        x="price",
        ax=ax2,
        color=COLORS["bars"]["purchase"],
        width=0.5,
    )

    ax2.set_title("Boxplot del precio", color=COLORS["title"])
    ax2.grid(True)

    plt.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2)


def plot_events_per_day(
    df: pd.DataFrame,
    output_path: Union[str, os.PathLike] = "events_per_day.png",
    show: bool = True,
    colors: dict | None = None,
):
    """
    Dibuja 3 paneles: stacked area por tipo + línea total, líneas por tipo (detalle)
    y sesiones únicas. Guarda la imagen en output_path si se proporciona.

    Parámetros
    ----------
    df : pd.DataFrame
        Debe contener: event_time, event_type, user_session
    output_path : str | PathLike
        Ruta para guardar la figura.
    show : bool
        Mostrar la figura al terminar.
    colors : dict | None
        Diccionario opcional con claves:
        'type_colors', 'total', 'session', 'bg', 'grid'
    Devuelve
    -------
    (fig, (ax1, ax2, ax3)), dict_aggregations
    """
    from pathlib import Path

    # Validaciones
    for col in ("event_time", "event_type", "user_session"):
        if col not in df.columns:
            raise ValueError(f"El DataFrame debe contener la columna '{col}'")

    df = df.copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])

    # Día del evento
    df["event_day"] = df["event_time"].dt.date

    # Aggregations
    events_per_day = df.groupby("event_day").size().rename("total")
    events_by_type = df.groupby(["event_day", "event_type"]).size().unstack(fill_value=0)
    sessions_per_day = df.groupby("event_day")["user_session"].nunique()

    # Palette por defecto
    default_TYPE_COLORS = {
        "view": "#378ADD",
        "cart": "#1D9E75",
        "purchase": "#D85A30",
        "remove_from_cart": "#D4537E",
    }
    defaults = {
        "type_colors": default_TYPE_COLORS,
        "total": "#2C2C2A",
        "session": "#7F77DD",
        "bg": "#FAFAF9",
        "grid": "#E8E6DF",
    }
    if colors is None:
        colors = defaults
    else:
        # completar valores faltantes
        merged = defaults.copy()
        merged.update(colors)
        colors = merged

    TYPE_COLORS = colors["type_colors"]
    TOTAL_COLOR = colors["total"]
    SESSION_COLOR = colors["session"]
    BG = colors["bg"]
    GRID = colors["grid"]

    # Preparar datos para plotting
    event_types = events_by_type.columns.tolist()
    colors_list = [TYPE_COLORS.get(t, "#B4B2A9") for t in event_types]
    x = pd.to_datetime(events_per_day.index)
    x_et = pd.to_datetime(events_by_type.index)

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.6, 1]})
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.08)

    # Panel 1: stacked area + total line
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    ax1.stackplot(x_et, [events_by_type[t] for t in event_types],
                  labels=event_types, colors=colors_list, alpha=0.72)
    ax1.plot(x, events_per_day.values, color=TOTAL_COLOR,
             linewidth=1.8, linestyle="--", label="Total", zorder=5)
    ax1.set_ylabel("Eventos", fontsize=10, color="#5F5E5A")
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"))
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.7,
               edgecolor=GRID, facecolor=BG, ncol=len(event_types) + 1)
    ax1.set_title("Actividad diaria — eventos & sesiones", fontsize=13,
                  pad=12, color="#2C2C2A", loc="left")
    ax1.grid(axis="y", color=GRID, linewidth=0.7)
    ax1.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax1.spines[sp].set_visible(False)

    # Panel 2: líneas por tipo
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    for t in event_types:
        ax2.plot(x_et, events_by_type[t],
                 color=TYPE_COLORS.get(t, "#B4B2A9"),
                 linewidth=1.6, label=t, marker="o",
                 markersize=2.5, markevery=3)
    ax2.set_ylabel("Eventos / tipo", fontsize=10, color="#5F5E5A")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"))
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.7,
               edgecolor=GRID, facecolor=BG, ncol=len(event_types))
    ax2.grid(axis="y", color=GRID, linewidth=0.7)
    ax2.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax2.spines[sp].set_visible(False)

    # Panel 3: sesiones únicas
    ax3 = axes[2]
    ax3.set_facecolor(BG)
    ax3.fill_between(pd.to_datetime(sessions_per_day.index),
                     sessions_per_day.values,
                     color=SESSION_COLOR, alpha=0.18)
    ax3.plot(pd.to_datetime(sessions_per_day.index),
             sessions_per_day.values,
             color=SESSION_COLOR, linewidth=1.8,
             marker="o", markersize=2.5, markevery=3,
             label="Sesiones únicas")
    ax3.set_ylabel("Sesiones", fontsize=10, color="#5F5E5A")
    ax3.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{int(v/1e3)}K"))
    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.7,
               edgecolor=GRID, facecolor=BG)
    ax3.grid(axis="y", color=GRID, linewidth=0.7)
    ax3.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax3.spines[sp].set_visible(False)

    # X-axis
    ax3.tick_params(axis="x", labelsize=9, labelcolor="#5F5E5A", rotation=20)

    # Guardar
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close(fig)

    aggregations = {
        "events_per_day": events_per_day,
        "events_by_type": events_by_type,
        "sessions_per_day": sessions_per_day,
    }
    return (fig, axes), aggregations


def plot_temporal_patterns(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "patrones_temporales.png"
):
    """
    Genera el Bloque 5 — Patrones temporales (1×3) siguiendo la estética original
    definida por el usuario. Guarda y/o muestra el gráfico según flags.
    """

    # ── Palette & style ───────────────────────────────────────────────────────
    BG      = '#FAFAF9'
    GRID    = '#E8E6DF'
    MUTED   = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def fmt_k(v, _):
        return f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v))

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    # ── Prep ──────────────────────────────────────────────────────────────────
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour']       = df['event_time'].dt.hour
    df['weekday']    = df['event_time'].dt.day_name()

    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_es    = ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo']

    pivot = df.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
    pivot = pivot.reindex(weekday_order)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig5, (ax_h, ax_w, ax_hm) = plt.subplots(1, 3, figsize=(15, 5))
    fig5.patch.set_facecolor(BG)
    fig5.suptitle(
        "Patrones temporales de actividad",
        fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left'
    )

    # ── Eventos por hora ──────────────────────────────────────────────────────
    hour_counts = df.groupby('hour').size()
    ax_h.bar(
        hour_counts.index, hour_counts.values,
        color=PALETTE[0], alpha=0.82,
        edgecolor='white', linewidth=0.4
    )
    ax_h.set_title("Eventos por hora del día", fontsize=10.5, color='#2C2C2A', pad=8)
    ax_h.set_xlabel("Hora", fontsize=9, color=MUTED)
    ax_h.set_ylabel("Eventos", fontsize=9, color=MUTED)
    ax_h.set_xticks(range(0, 24, 3))
    ax_h.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    style_ax(ax_h)

    # ── Eventos por día de la semana ─────────────────────────────────────────
    wd_counts = df.groupby('weekday').size().reindex(weekday_order)
    ax_w.bar(
        range(7), wd_counts.values,
        color=PALETTE[1], alpha=0.82,
        edgecolor='white', linewidth=0.4
    )
    ax_w.set_title("Eventos por día de la semana", fontsize=10.5, color='#2C2C2A', pad=8)
    ax_w.set_xticks(range(7))
    ax_w.set_xticklabels(weekday_es, fontsize=8, rotation=25, ha='right')
    ax_w.set_ylabel("Eventos", fontsize=9, color=MUTED)
    ax_w.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))
    style_ax(ax_w)

    # ── Heatmap día × hora ───────────────────────────────────────────────────
    im = ax_hm.imshow(
        pivot.values, aspect='auto',
        cmap='Blues', interpolation='nearest'
    )
    ax_hm.set_title("Heatmap día × hora", fontsize=10.5, color='#2C2C2A', pad=8)
    ax_hm.set_yticks(range(7))
    ax_hm.set_yticklabels(weekday_es, fontsize=8, color=MUTED)
    ax_hm.set_xticks(range(0, 24, 3))
    ax_hm.set_xticklabels(range(0, 24, 3), fontsize=8, color=MUTED)
    ax_hm.set_xlabel("Hora", fontsize=9, color=MUTED)

    fig5.colorbar(im, ax=ax_hm, shrink=0.8, label='Eventos')

    ax_hm.set_facecolor(BG)
    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    # ── Show or close ─────────────────────────────────────────────────────────
    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}\n")


def plot_top_middle_tail(df):
    """
    Visualiza las categorías Top, Middle y Tail siguiendo el estilo TFM.
    - Top 4 categorías más frecuentes
    - Middle 4 categorías del centro de la distribución
    - Tail 10 categorías menos frecuentes
    """
    if "category_code" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'category_code'")

    counts = df["category_code"].value_counts()
    if counts.empty:
        raise ValueError("No hay categorías en el DataFrame")

    counts_df = counts.rename_axis("category").reset_index(name="count")

    top4 = counts_df.head(4).assign(group="Top 4")
    middle_start = max((len(counts_df) - 4) // 2, 0)
    middle4 = counts_df.iloc[middle_start: middle_start + 4].assign(group="Middle 4")
    tail10 = counts_df.tail(10).assign(group="Tail 10")

    plot_df = pd.concat([top4, middle4, tail10], ignore_index=True)

    group_colors = {
        "Top 4": COLORS["bars"]["view"],
        "Middle 4": COLORS["bars"]["cart"],
        "Tail 10": COLORS["bars"]["purchase"],
    }

    plt.figure(figsize=(11, 7))
    ax = sns.barplot(
        data=plot_df,
        y="category",
        x="count",
        hue="group",
        palette=group_colors,
        dodge=False,
        alpha=0.85,
    )

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else str(v)
    ))

    plt.title("Top, Middle y Tail categorías por número de eventos", color=COLORS["title"])
    plt.xlabel("Número de eventos")
    plt.ylabel("Categoría")

    plt.grid(axis="x")
    plt.legend(title="", framealpha=0.6)
    plt.tight_layout()
    plt.show()

    return plot_df


def plot_price_distribution_top10(
    df,
    output_path: Union[str, os.PathLike] = "price_distribution_top10.png",
    show: bool = True,
):
    """
    Visualiza la distribución del precio para las Top 10 categorías
    siguiendo el estilo TFM definido en apply_style().
    """
    COLORS = apply_style()
    BG = COLORS.get("bg", "#FAFAF9")

    top_cats = df["category_code"].value_counts().head(10)
    top_cat_list = top_cats.index.tolist()
    df_top = df[df["category_code"].isin(top_cat_list)]

    palette = [
        COLORS["bars"]["view"],
        COLORS["bars"]["cart"],
        COLORS["bars"]["purchase"],
        COLORS["lines"]["ratio_view_cart"],
        COLORS["lines"]["ratio_cart_purchase"],
        "#8FB3D9",
        "#7AAE92",
        "#E39A6A",
        "#A89CE3",
        "#E58AA8",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(
        data=df_top,
        x="price",
        hue="category_code",
        bins=40,
        palette=palette,
        alpha=0.45,
        ax=ax,
        element="step",
        multiple="layer",
        legend=True,
    )

    ax.set_title("Distribución del precio por categoría (Top 10)", color=COLORS["title"])
    ax.set_xlabel("Precio")
    ax.set_ylabel("Frecuencia")

    ax.grid(axis="y")

    legend_handles = [
        Patch(facecolor=palette[i], alpha=0.45, label=cat)
        for i, cat in enumerate(top_cat_list)
    ]
    ax.legend(handles=legend_handles, title="Categoría",
              framealpha=0.6, loc="upper right")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_top


def plot_events_by_category_top10(
    df_top: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "events_by_category.png"
):
    """
    Genera un gráfico de líneas + área para los eventos diarios por categoría (Top 10),
    siguiendo el estilo definido en el snippet del usuario.

    Parámetros
    ----------
    df_top : pd.DataFrame
        DataFrame filtrado con las Top categorías.
    show : bool, opcional
        Si True, muestra el gráfico. Si False, solo guarda la imagen.
    output_path : Union[str, os.PathLike], opcional
        Ruta donde guardar el gráfico.
    """

    # --- Estilo definido por el usuario ---
    BG     = '#FAFAF9'
    GRID   = '#E8E6DF'
    MUTED  = '#888780'

    PALETTE = [
        '#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
        '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11',
    ]

    # --- Prep ---
    events_cat_day = (
        df_top.groupby(['event_day', 'category_code'])
              .size()
              .reset_index(name='events')
    )
    events_cat_day['event_day'] = pd.to_datetime(events_cat_day['event_day'])

    categories = events_cat_day['category_code'].unique()
    color_map  = {cat: PALETTE[i % len(PALETTE)] for i, cat in enumerate(categories)}

    # Short labels (last segment of dotted category name)
    def short_label(cat):
        return cat.split('.')[-1].replace('_', ' ').title() if isinstance(cat, str) else str(cat)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    lines = []
    for cat in categories:
        sub = events_cat_day[events_cat_day['category_code'] == cat].sort_values('event_day')

        line, = ax.plot(
            sub['event_day'], sub['events'],
            color=color_map[cat], linewidth=1.8,
            marker='o', markersize=3, markevery=3,
            alpha=0.9, zorder=3, label=short_label(cat)
        )

        ax.fill_between(
            sub['event_day'], sub['events'],
            color=color_map[cat], alpha=0.06
        )

        lines.append(line)

    # --- Axes ---
    ax.set_title(
        "Eventos por día y categoría  —  Top 10",
        fontsize=13, pad=14, color='#2C2C2A', loc='left'
    )

    ax.set_xlabel("")
    ax.set_ylabel("Eventos", fontsize=10, color=MUTED)

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(
            lambda v, _: (
                f'{v/1e6:.1f}M' if v >= 1e6 else
                f'{int(v/1e3)}K' if v >= 1e3 else
                str(int(v))
            )
        )
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))

    ax.tick_params(axis='x', labelsize=9, labelcolor=MUTED, rotation=20)
    ax.tick_params(axis='y', labelsize=9, labelcolor=MUTED)

    ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)

    ax.legend(
        handles=lines,
        labels=[short_label(c) for c in categories],
        loc='upper left', bbox_to_anchor=(1.01, 1),
        fontsize=8.5, framealpha=0.8,
        edgecolor=GRID, facecolor=BG,
        title='Categoría', title_fontsize=9,
        borderpad=0.8, labelspacing=0.6
    )

    plt.tight_layout()

    # --- Save ---
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    # --- Show or close ---
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Guardado como {output_path}")


def plot_activity_engagement(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "actividad_engagement_por_usuario.png"
):
    """
    Genera el bloque 1 — Actividad y engagement (2×2) siguiendo el estilo definido
    en el snippet original del usuario. Guarda y/o muestra el gráfico según flags.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas: user_id, user_session, event_time.
    show : bool, opcional
        Si True, muestra el gráfico. Si False, solo guarda la imagen.
    output_path : Union[str, os.PathLike], opcional
        Ruta donde guardar el gráfico.
    """

    # ── Palette & style ───────────────────────────────────────────────────────
    BG      = '#FAFAF9'
    GRID    = '#E8E6DF'
    MUTED   = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def fmt_k(v, _):
        return f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v))

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    # ── Prep ──────────────────────────────────────────────────────────────────
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['event_day']  = df['event_time'].dt.date

    # ── Métricas ──────────────────────────────────────────────────────────────
    sessions_per_user = df.groupby('user_id')['user_session'].nunique()
    active_days       = df.groupby('user_id')['event_day'].nunique()

    session_duration = (
        df.groupby(['user_id', 'user_session'])['event_time']
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
        .reset_index(name='duration_min')
    )
    avg_duration = session_duration.groupby('user_id')['duration_min'].mean()

    session_ts = (
        df.groupby(['user_id', 'user_session'])['event_time']
        .min().reset_index().sort_values(['user_id', 'event_time'])
    )
    session_ts['gap_days'] = (
        session_ts.groupby('user_id')['event_time']
        .diff().dt.total_seconds() / 86400
    )
    avg_gap = session_ts.groupby('user_id')['gap_days'].mean().dropna()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig1.patch.set_facecolor(BG)
    fig1.suptitle(
        "Actividad y engagement por usuario",
        fontsize=14, color='#2C2C2A', y=1.01, x=0.02, ha='left'
    )

    datasets = [
        (axes[0,0], sessions_per_user, "Sesiones únicas por usuario",  "Sesiones",  PALETTE[0]),
        (axes[0,1], active_days,       "Días activos por usuario",      "Días",      PALETTE[1]),
        (axes[1,0], avg_duration,      "Duración media de sesión (min)","Minutos",   PALETTE[2]),
        (axes[1,1], avg_gap,           "Gap entre sesiones (días)",     "Días",      PALETTE[3]),
    ]

    for ax, data, title, xlabel, color in datasets:
        clipped = data.clip(upper=data.quantile(0.99))
        ax.hist(
            clipped, bins=40,
            color=color, alpha=0.82,
            edgecolor='white', linewidth=0.4
        )
        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.set_ylabel("Usuarios", fontsize=9, color=MUTED)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))

        med = data.median()
        ax.axvline(med, color=color, linewidth=1.5, linestyle='--', alpha=0.8)
        ax.text(
            med * 1.03, ax.get_ylim()[1] * 0.88,
            f'mediana\n{med:.1f}',
            fontsize=7.5, color=color
        )

        style_ax(ax)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    # ── Show or close ─────────────────────────────────────────────────────────
    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}\n")


def plot_funnel_conversion(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "funnel_conversion.png"
):
    """
    Genera el funnel de conversión por usuario siguiendo el estilo definido
    en el snippet original del usuario. Guarda y/o muestra el gráfico según flags.
    """

    # ── Palette & style ───────────────────────────────────────────────────────
    BG      = '#FAFAF9'
    GRID    = '#E8E6DF'
    MUTED   = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    # ── Funnel metrics ───────────────────────────────────────────────────────
    user_events = df.groupby('user_id')['event_type'].apply(set)
    n_view     = (user_events.apply(lambda s: 'view'     in s)).sum()
    n_cart     = (user_events.apply(lambda s: 'cart'     in s)).sum()
    n_purchase = (user_events.apply(lambda s: 'purchase' in s)).sum()
    total      = len(user_events)

    stages  = ['Todos\nlos usuarios', 'Al menos\n1 view', 'Al menos\n1 carrito', 'Al menos\n1 compra']
    counts  = [total, n_view, n_cart, n_purchase]
    colors  = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]]
    pcts    = [100] + [c / total * 100 for c in counts[1:]]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(10, 5.5))
    fig2.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.set_title(
        "Funnel de conversión por usuario",
        fontsize=13, color='#2C2C2A', pad=14, loc='left'
    )

    bar_h   = 0.13
    gap     = 0.04
    widths  = [pcts[i] / 100 for i in range(4)]
    tops    = [0.85 - i * (bar_h + gap) for i in range(4)]

    for i, (stage, count, pct, color, w, top) in enumerate(
            zip(stages, counts, pcts, colors, widths, tops)):

        left = (1 - w) / 2

        ax.add_patch(plt.Rectangle(
            (left, top), w, bar_h,
            color=color, alpha=0.82, zorder=3
        ))

        ax.text(
            0.5, top + bar_h / 2,
            f'{count:,.0f} usuarios  ({pct:.1f}%)',
            ha='center', va='center',
            fontsize=10, color='white', fontweight='500', zorder=4
        )

        ax.text(
            left - 0.02, top + bar_h / 2,
            stage,
            ha='right', va='center',
            fontsize=8.5, color=MUTED
        )

        if i > 0:
            drop = (1 - counts[i] / counts[i-1]) * 100
            ax.text(
                0.5, top + bar_h + gap / 2,
                f'↓ {drop:.1f}% no continúa',
                ha='center', va='center',
                fontsize=7.5, color=MUTED
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}\n")


def plot_economic_value(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "valor_economico.png"
):
    """
    Genera el bloque 3 — Valor económico por usuario (1×3) siguiendo el estilo
    definido en el snippet original del usuario. Guarda y/o muestra el gráfico.
    """

    # ── Palette & style ───────────────────────────────────────────────────────
    BG      = '#FAFAF9'
    GRID    = '#E8E6DF'
    MUTED   = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def fmt_k(v, _):
        return f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v))

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    # ── Economic metrics ──────────────────────────────────────────────────────
    purchases = df[df['event_type'] == 'purchase']
    spend     = purchases.groupby('user_id')['price'].sum()
    ticket    = purchases.groupby('user_id')['price'].mean()
    freq      = purchases.groupby('user_id').size()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig3, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig3.patch.set_facecolor(BG)
    fig3.suptitle(
        "Valor económico por usuario",
        fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left'
    )

    eco_data = [
        (axes[0], spend,   "Gasto total por usuario",       "€",         PALETTE[0]),
        (axes[1], ticket,  "Ticket medio por usuario",       "€ / compra",PALETTE[1]),
        (axes[2], freq,    "Nº de compras por usuario",      "Compras",   PALETTE[2]),
    ]

    for ax, data, title, xlabel, color in eco_data:
        clipped = data.clip(upper=data.quantile(0.99))

        ax.hist(
            clipped, bins=40,
            color=color, alpha=0.82,
            edgecolor='white', linewidth=0.4
        )

        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
        ax.set_ylabel("Usuarios", fontsize=9, color=MUTED)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_k))

        p50  = data.median()
        p80  = data.quantile(0.8)

        ax.axvline(p50, color=color, linewidth=1.4, linestyle='--', alpha=0.8)
        ax.axvline(p80, color=color, linewidth=1.4, linestyle=':',  alpha=0.6)

        ymax = ax.get_ylim()[1]
        ax.text(p50 * 1.03, ymax * 0.88, f'p50\n{p50:.0f}', fontsize=7.5, color=color)
        ax.text(p80 * 1.03, ymax * 0.65, f'p80\n{p80:.0f}', fontsize=7.5, color=color, alpha=0.8)

        style_ax(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}\n")


def plot_corr_user_metrics(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "corr_heatmap_usuarios.png"
):
    BG     = '#FAFAF9'
    GRID   = '#E8E6DF'
    MUTED  = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def fmt_k(v, _):
        return f'{v/1e6:.1f}M' if v >= 1e6 else f'{int(v/1e3)}K' if v >= 1e3 else str(int(v))

    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    purchases = df[df['event_type'] == 'purchase']

    session_dur = (
        df.groupby(['user_id', 'user_session'])['event_time']
        .agg(lambda x: (x.max() - x.min()).total_seconds() / 60)
        .groupby('user_id').mean()
        .rename('avg_session_min')
    )

    user_metrics = pd.DataFrame({
        'n_events'      : df.groupby('user_id').size(),
        'n_sessions'    : df.groupby('user_id')['user_session'].nunique(),
        'active_days'   : df.groupby('user_id')['event_day'].nunique(),
        'avg_session_min': session_dur,
        'total_spend'   : purchases.groupby('user_id')['price'].sum(),
        'avg_ticket'    : purchases.groupby('user_id')['price'].mean(),
        'n_purchases'   : purchases.groupby('user_id').size(),
    }).dropna()

    user_clip = user_metrics.apply(lambda c: c.clip(upper=c.quantile(0.99)))
    corr = user_clip.corr()

    labels = ['Eventos', 'Sesiones', 'Días activos', 'Duración\nsesión (min)',
              'Gasto total', 'Ticket medio', 'Nº compras']

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5, color=MUTED, rotation=30, ha='right')
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8.5, color=MUTED)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(v) > 0.5 else '#2C2C2A')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Correlación de Pearson')
    ax.set_title("Correlación entre métricas de usuario", fontsize=13,
                 pad=14, color='#2C2C2A', loc='left')

    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(labelcolor=MUTED)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}")


def plot_price_vs_behavior(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "corr_precio_comportamiento.png"
):
    BG     = '#FAFAF9'
    GRID   = '#E8E6DF'
    MUTED  = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    def add_corr_label(ax, x, y):
        mask = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.text(0.97, 0.05, f'r = {r:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color=MUTED,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG,
                          edgecolor=GRID, alpha=0.8))

    df = df.copy()
    purchases = df[df['event_type'] == 'purchase']

    product_stats = (
        df.groupby('product_id')
        .agg(
            views     = ('event_type', lambda x: (x == 'view').sum()),
            carts     = ('event_type', lambda x: (x == 'cart').sum()),
            purchases = ('event_type', lambda x: (x == 'purchase').sum()),
            avg_price = ('price', 'mean'),
        )
        .query('views >= 10')
    )

    product_stats['conv_view_purchase'] = product_stats['purchases'] / product_stats['views']
    product_stats['conv_view_cart']     = product_stats['carts']     / product_stats['views']
    product_stats['abandon_rate']       = 1 - (product_stats['purchases'] /
                                               product_stats['carts'].replace(0, np.nan))

    ps = product_stats.dropna().apply(lambda c: c.clip(upper=c.quantile(0.99)) if c.dtype == float else c)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Precio vs comportamiento de compra", fontsize=14,
                 color='#2C2C2A', y=1.01, x=0.02, ha='left')

    scatters = [
        (axes[0,0], ps['avg_price'], ps['conv_view_purchase'],
         'Precio medio (€)', 'Conv. view→purchase', PALETTE[0],
         'Precio vs conversión view→compra'),
        (axes[0,1], ps['avg_price'], ps['conv_view_cart'],
         'Precio medio (€)', 'Conv. view→carrito', PALETTE[1],
         'Precio vs conversión view→carrito'),
        (axes[1,0], ps['avg_price'], ps['abandon_rate'],
         'Precio medio (€)', 'Tasa abandono carrito', PALETTE[2],
         'Precio vs abandono de carrito'),
        (axes[1,1], ps['avg_price'], ps['purchases'],
         'Precio medio (€)', 'Total compras', PALETTE[3],
         'Precio vs frecuencia de compra'),
    ]

    for ax, x, y, xl, yl, color, title in scatters:
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], alpha=0.18, s=8, color=color, zorder=3)

        z = np.polyfit(x[mask], y[mask], 1)
        xr = np.linspace(x[mask].min(), x[mask].max(), 200)
        ax.plot(xr, np.poly1d(z)(xr), color=color, linewidth=1.8, zorder=4)

        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.set_xlabel(xl, fontsize=9, color=MUTED)
        ax.set_ylabel(yl, fontsize=9, color=MUTED)

        add_corr_label(ax, x[mask].values, y[mask].values)
        style_ax(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}")


def plot_category_ratios(
    df: pd.DataFrame,
    show: bool = True,
    output_path: Union[str, os.PathLike] = "corr_ratios_categoria.png"
):
    """
    Genera el Bloque 3 — Ratios por categoría (1×3) siguiendo la estética original.
    Guarda y/o muestra el gráfico según flags.
    """

    # ── Style ───────────────────────────────────────────────────────────────
    BG     = '#FAFAF9'
    GRID   = '#E8E6DF'
    MUTED  = '#888780'
    PALETTE = ['#378ADD', '#1D9E75', '#D85A30', '#7F77DD', '#D4537E',
               '#BA7517', '#0F6E56', '#993C1D', '#185FA5', '#3B6D11']

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.grid(axis='y', color=GRID, linewidth=0.7, linestyle='--')
        ax.set_axisbelow(True)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    # ── Prep ───────────────────────────────────────────────────────────────
    df = df.copy()

    cat_stats = (
        df.groupby('category_code')
        .agg(
            views     = ('event_type', lambda x: (x == 'view').sum()),
            carts     = ('event_type', lambda x: (x == 'cart').sum()),
            purchases = ('event_type', lambda x: (x == 'purchase').sum()),
            avg_price = ('price', 'mean'),
        )
        .query('views >= 100')
        .assign(
            ratio_vc = lambda d: d['carts']     / d['views'] * 100,
            ratio_cp = lambda d: d['purchases'] / d['carts'] * 100,
            avg_ticket = lambda d: d['avg_price'],
        )
        .dropna()
        .nlargest(15, 'views')
        .sort_values('ratio_vc')
    )

    cats = [c.split('.')[-1].replace('_', ' ').title() for c in cat_stats.index]
    y    = range(len(cats))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        "Conversión y precio por categoría (Top 15 por volumen)",
        fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left'
    )

    plots = [
        (axes[0], 'ratio_vc',  PALETTE[0], 'Ratio view → carrito (%)',    '%'),
        (axes[1], 'ratio_cp',  PALETTE[1], 'Ratio carrito → compra (%)',  '%'),
        (axes[2], 'avg_ticket',PALETTE[2], 'Precio medio (€)',            '€'),
    ]

    for ax, col, color, title, xlabel in plots:
        vals = cat_stats[col].values

        bars = ax.barh(
            list(y), vals,
            color=color, alpha=0.82,
            edgecolor='white', linewidth=0.4
        )

        ax.set_yticks(list(y))
        ax.set_yticklabels(cats, fontsize=8.5)
        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_width() + vals.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}',
                va='center', fontsize=7.5, color=MUTED
            )

        style_ax(ax)
        ax.grid(axis='x', color=GRID, linewidth=0.7, linestyle='--')
        ax.grid(axis='y', visible=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"   Guardado: {output_path}")
