import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy import stats
from typing import Union, Optional
import os
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
BG      = '#FAFAF9'
GRID    = '#E8E6DF'
MUTED   = '#888780'
PALETTE = ['#378ADD','#1D9E75','#D85A30','#7F77DD','#D4537E',
           '#BA7517','#0F6E56','#993C1D','#185FA5','#3B6D11']
AGGREGATION_TYPE = "generic"


def style_ax(ax, grid_axis='y'):
    ax.set_facecolor(BG)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, linestyle='--')
    ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(labelcolor=MUTED, labelsize=9)

def short(cat):
    return str(cat).split('.')[-1].replace('_',' ').title()

fmt_k = lambda v, _: f'{v/1e6:.1f}M' if v>=1e6 else f'{int(v/1e3)}K' if v>=1e3 else f'{v:.1f}'


def plot_distributions(
    df,
    output_path: Union[str, os.PathLike] = f"distributions_{AGGREGATION_TYPE}.png",
):
    df['short'] = df['category_code'].apply(short)

    df['visits_per_user']  = df['visits']  / df['users']
    df['clicks_per_user']  = df['clicks']  / df['users']
    df['clicks_per_visit'] = df['clicks']  / df['visits']
    df['dwell_skew']       = df['dwelltime_l'] / df['dwell_time_avg']

    FEAT_COLS   = ['visits_per_user','clicks_per_user','click_depth',
                   'dwell_time_avg','active_days','return_rate','dwell_skew']
    FEAT_LABELS = ['Visitas/user','Clics/user','Click depth',
                   'Dwell avg','Días activos','Return rate','Dwell skew']

    if output_path is None:
        output_path = f"generic_01_distribuciones.png"

    fig, axes = plt.subplots(2, len(FEAT_COLS), figsize=(20, 7))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Distribución de métricas de engagement (128 categorías)",
                 fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left')

    for j, (col, label, color) in enumerate(zip(FEAT_COLS, FEAT_LABELS, PALETTE)):
        data = df[col].dropna()
        clipped = data.clip(upper=data.quantile(0.99))

        ax = axes[0, j]
        ax.hist(clipped, bins=25, color=color, alpha=0.82,
                edgecolor='white', linewidth=0.4)
        p50 = data.median()
        ax.axvline(p50, color=color, linewidth=1.5, linestyle='--')
        ax.set_title(label, fontsize=9, color='#2C2C2A', pad=6)
        ax.set_ylabel("Categorías" if j==0 else "", fontsize=8, color=MUTED)
        style_ax(ax)

        ax = axes[1, j]
        ax.boxplot(clipped, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.5, linewidth=0),
                   medianprops=dict(color=color, linewidth=2),
                   whiskerprops=dict(color=MUTED, linewidth=1),
                   capprops=dict(color=MUTED, linewidth=1),
                   flierprops=dict(marker='o', color=color, alpha=0.4,
                                   markersize=4, linestyle='none'))
        ax.set_xticklabels([])
        style_ax(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    return fig


def plot_top_bottom(
    df,
    output_path: Union[str, os.PathLike] = f"top_bottom_{AGGREGATION_TYPE}.png",
):
    if output_path is None:
        output_path = f"generic_02_top_bottom.png"

    metrics_tb = [
        ('dwell_time_avg', 'Dwell time avg (s)',  PALETTE[0]),
        ('return_rate',    'Return rate',          PALETTE[1]),
        ('clicks_per_visit','Clics / visita',      PALETTE[2]),
        ('click_depth',    'Click depth',          PALETTE[3]),
    ]

    fig, axes = plt.subplots(len(metrics_tb), 2, figsize=(16, 4*len(metrics_tb)))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Top / Bottom 10 categorías por métrica",
                 fontsize=14, color='#2C2C2A', y=1.01, x=0.02, ha='left')

    for i, (col, label, color) in enumerate(metrics_tb):
        top10 = df.nlargest(10, col)[['category_code', col]].sort_values(col)
        bot10 = df.nsmallest(10, col)[['category_code', col]].sort_values(col, ascending=False)

        for j, (data, title, alpha) in enumerate([
            (top10, f'Top 10 — {label}', 0.85),
            (bot10, f'Bottom 10 — {label}', 0.45),
        ]):
            ax = axes[i, j]
            bars = ax.barh(data['category_code'], data[col], color=color,
                           alpha=alpha, edgecolor='white', linewidth=0.4)
            for bar, val in zip(bars, data[col]):
                ax.text(bar.get_width() + data[col].max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}', va='center', fontsize=7.5, color=MUTED)
            ax.set_title(title, fontsize=10, color='#2C2C2A', pad=6)
            style_ax(ax, grid_axis='x')
            ax.grid(axis='y', visible=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    return fig


def plot_correlations(
    df,
    output_path: Union[str, os.PathLike] = f"correlations_{AGGREGATION_TYPE}.png"
):
    df['visits_per_user']  = df['visits']  / df['users']
    df['clicks_per_user']  = df['clicks']  / df['users']
    df['clicks_per_visit'] = df['clicks']  / df['visits']
    df['dwell_skew']       = df['dwelltime_l'] / df['dwell_time_avg']

    FEAT_COLS   = ['visits_per_user','clicks_per_user','click_depth',
                   'dwell_time_avg','active_days','return_rate','dwell_skew']
    FEAT_LABELS = ['Visitas/user','Clics/user','Click depth',
                   'Dwell avg','Días activos','Return rate','Dwell skew']

    if output_path is None:
        output_path = f"generic_04_correlaciones.png"

    corr = df[FEAT_COLS].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(FEAT_LABELS)))
    ax.set_xticklabels(FEAT_LABELS, fontsize=9, color=MUTED, rotation=35, ha='right')
    ax.set_yticks(range(len(FEAT_LABELS)))
    ax.set_yticklabels(FEAT_LABELS, fontsize=9, color=MUTED)

    for i in range(len(FEAT_LABELS)):
        for j in range(len(FEAT_LABELS)):
            v = corr.values[i,j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=8.5, color='white' if abs(v) > 0.5 else '#2C2C2A')

    fig.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')
    ax.set_title("Correlación entre métricas de categoría",
                 fontsize=13, pad=14, color='#2C2C2A', loc='left')

    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(labelcolor=MUTED)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.show()
    return fig
