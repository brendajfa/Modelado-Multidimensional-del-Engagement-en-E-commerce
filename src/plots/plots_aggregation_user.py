import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from typing import Union
import os
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
BG      = '#FAFAF9'
GRID    = '#E8E6DF'
MUTED   = '#888780'
SEG_COLORS = {
    'Tourist':    '#B4C9E8',
    'Interested': '#378ADD',
    'Average':    '#1D9E75',
    'Active':     '#D85A30',
    'VIP':        '#7F77DD',
}
SEGMENTS = ['Tourist', 'Interested', 'Average', 'Active', 'VIP']
PALETTE  = list(SEG_COLORS.values())
AGGREGATION_TYPE = "user"


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


def plot_funnel_composition(
    df,
    output_path: Union[str, os.PathLike] = f"funnel_composicion_{AGGREGATION_TYPE}.png"
):
    df_sorted = df.sort_values('pct_VIP', ascending=False)

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Composición de usuarios por segmento (Tourist → VIP)",
                 fontsize=14, color='#2C2C2A', y=1.01, x=0.02, ha='left')

    for ax, data, title in [
        (ax_full, df_sorted, "Todas las categorías — ordenadas por % VIP"),
        (ax_zoom, df_sorted.head(30), "Top 30 categorías con mayor % VIP"),
    ]:
        bottom = np.zeros(len(data))
        x = np.arange(len(data))
        for s in SEGMENTS:
            vals = data[f'pct_{s}'].values
            ax.bar(x, vals, bottom=bottom, color=SEG_COLORS[s],
                   label=s, width=0.8, edgecolor='none')
            bottom += vals

        ax.set_xlim(-0.5, len(data)-0.5)
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(data['short'], rotation=70, ha='right', fontsize=6.5)
        ax.set_ylabel("% usuarios", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.tick_params(labelcolor=MUTED, labelsize=7)
        ax.set_facecolor(BG)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f'{v:.0f}%'))

    handles = [mpatches.Patch(color=SEG_COLORS[s], label=s) for s in SEGMENTS]
    ax_full.legend(handles=handles, loc='upper right', fontsize=9,
                   framealpha=0.8, edgecolor=GRID, facecolor=BG)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_conversion_ratios(
    df,
    output_path: Union[str, os.PathLike] = f"conversion_ratios_{AGGREGATION_TYPE}.png"
):
    ratio_cols  = [f'ratio_{s}' for s in SEGMENTS[1:]]
    ratio_labels = [f'Tourist→{s}' for s in SEGMENTS[1:]]

    fig, axes = plt.subplots(1, len(ratio_cols), figsize=(18, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Tasa de conversión por segmento relativa a Tourist",
                 fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left')

    for ax, col, label, color in zip(axes, ratio_cols, ratio_labels, PALETTE[1:]):
        data = df[col].dropna()
        clipped = data.clip(upper=data.quantile(0.99))
        ax.hist(clipped, bins=25, color=color, alpha=0.82,
                edgecolor='white', linewidth=0.4)
        p50 = data.median(); p90 = data.quantile(0.9)
        ax.axvline(p50, color=color, linewidth=1.5, linestyle='--')
        ax.axvline(p90, color=color, linewidth=1.2, linestyle=':')
        ax.text(p50, ax.get_ylim()[1]*0.92, f'p50\n{p50:.3f}',
                fontsize=7, color=color, ha='left')
        ax.set_title(label, fontsize=10, color='#2C2C2A', pad=8)
        ax.set_xlabel("Ratio (usuarios seg / Tourist)", fontsize=8, color=MUTED)
        ax.set_ylabel("Categorías" if ax==axes[0] else "", fontsize=8, color=MUTED)
        style_ax(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_top_bottom_conversion(
    df,
    output_path: Union[str, os.PathLike] = f"top_bottom_conversion_{AGGREGATION_TYPE}.png"
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Top / Bottom 15 categorías — conversión a VIP y Active",
                 fontsize=14, color='#2C2C2A', y=1.01, x=0.02, ha='left')

    for i, (col, label, color) in enumerate([
        ('ratio_VIP',    'Ratio Tourist→VIP',    SEG_COLORS['VIP']),
        ('ratio_Active', 'Ratio Tourist→Active', SEG_COLORS['Active']),
    ]):
        top15 = df.nlargest(15, col)[['short', col]].sort_values(col)
        bot15 = df[df[col]>0].nsmallest(15, col)[['short', col]].sort_values(col, ascending=False)

        for j, (data, title, alpha) in enumerate([
            (top15, f'Top 15 — {label}',    0.85),
            (bot15, f'Bottom 15 — {label}', 0.40),
        ]):
            ax = axes[i, j]
            bars = ax.barh(data['short'], data[col], color=color,
                           alpha=alpha, edgecolor='white', linewidth=0.4)
            for bar, val in zip(bars, data[col]):
                ax.text(bar.get_width() + data[col].max()*0.01,
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.4f}', va='center', fontsize=7.5, color=MUTED)
            ax.set_title(title, fontsize=10, color='#2C2C2A', pad=6)
            style_ax(ax, grid_axis='x')
            ax.grid(axis='y', visible=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_heatmap_click_depth(
    df,
    output_path: Union[str, os.PathLike] = f"heatmap_click_depth_{AGGREGATION_TYPE}.png"
):
    depth_cols = [f'{s}_click_depth' for s in SEGMENTS]
    top40 = df.nlargest(40, 'total_users')
    heat_data = top40[depth_cols].copy()
    heat_data.columns = SEGMENTS
    heat_data.index   = top40['short']

    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor(BG)
    im = ax.imshow(heat_data.values, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')
    ax.set_xticks(range(len(SEGMENTS)))
    ax.set_xticklabels(SEGMENTS, fontsize=10, color=MUTED)
    ax.set_yticks(range(len(top40)))
    ax.set_yticklabels(heat_data.index, fontsize=8, color=MUTED)

    for i in range(len(top40)):
        for j in range(len(SEGMENTS)):
            v = heat_data.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if v > heat_data.values.max()*0.7 else '#2C2C2A')

    fig.colorbar(im, ax=ax, shrink=0.5, label='Click depth')
    ax.set_title("Click depth por segmento — Top 40 categorías por volumen",
                 fontsize=12, pad=14, color='#2C2C2A', loc='left')
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(labelcolor=MUTED)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_engagement_distributions(
    long,
    output_path: Union[str, os.PathLike] = f"engagement_segmentos_{AGGREGATION_TYPE}.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Engagement por segmento — distribución entre categorías",
                 fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left')

    for ax, metric, title, ylabel in [
        (axes[0], 'visits_per_user', "Visitas / usuario por segmento",   "Visitas / usuario"),
        (axes[1], 'clicks_per_visit','Clics / visita por segmento',      "Clics / visita"),
    ]:
        data_by_seg = [
            long[long['segment']==s][metric].dropna().clip(
                upper=long[long['segment']==s][metric].quantile(0.99)
            )
            for s in SEGMENTS
        ]

        bp = ax.boxplot(data_by_seg, patch_artist=True, notch=False,
                        medianprops=dict(linewidth=2),
                        whiskerprops=dict(color=MUTED, linewidth=1),
                        capprops=dict(color=MUTED, linewidth=1),
                        flierprops=dict(marker='o', markersize=3,
                                       linestyle='none', alpha=0.3))
        for patch, s in zip(bp['boxes'], SEGMENTS):
            patch.set_facecolor(SEG_COLORS[s])
            patch.set_alpha(0.75)
            patch.set_linewidth(0)
        for median, s in zip(bp['medians'], SEGMENTS):
            median.set_color(SEG_COLORS[s])

        ax.set_xticklabels(SEGMENTS, fontsize=9)
        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
        style_ax(ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_retention_vs_volume(
    df,
    output_path: Union[str, os.PathLike] = f"scatter_retencion_{AGGREGATION_TYPE}.png",
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Retención vs volumen de usuarios",
                 fontsize=14, color='#2C2C2A', y=1.02, x=0.02, ha='left')

    for ax, x_col, y_col, x_lab, y_lab, color, title in [
        (ax1, 'total_users', 'ratio_VIP',
         'Total usuarios (log)', 'Ratio Tourist→VIP',
         SEG_COLORS['VIP'], 'Volumen vs conversión a VIP'),
        (ax2, 'ratio_Active', 'ratio_VIP',
         'Ratio Tourist→Active', 'Ratio Tourist→VIP',
         SEG_COLORS['Active'], 'Retención Active vs VIP'),
    ]:
        x = df[x_col].clip(upper=df[x_col].quantile(0.99))
        y = df[y_col].clip(upper=df[y_col].quantile(0.99))

        if x_col == 'total_users':
            ax.scatter(np.log10(x+1), y, color=color, alpha=0.45, s=35, zorder=3)
            for _, row in df.nlargest(5, 'ratio_VIP').iterrows():
                ax.annotate(row['short'],
                            (np.log10(row['total_users']+1), row['ratio_VIP']),
                            fontsize=7, color=color,
                            xytext=(4,3), textcoords='offset points')
            ax.set_xlabel(x_lab, fontsize=9, color=MUTED)
            xv = np.log10(x+1).values; yv = y.values
            mask = np.isfinite(xv) & np.isfinite(yv)
            z = np.polyfit(xv[mask], yv[mask], 1)
            xr = np.linspace(xv[mask].min(), xv[mask].max(), 200)
            ax.plot(xr, np.poly1d(z)(xr), color=color, linewidth=1.8, alpha=0.6)
            r = np.corrcoef(xv[mask], yv[mask])[0,1]
        else:
            ax.scatter(x, y, color=color, alpha=0.45, s=35, zorder=3)
            for _, row in df.nlargest(5, 'ratio_VIP').iterrows():
                ax.annotate(row['short'],
                            (row['ratio_Active'], row['ratio_VIP']),
                            fontsize=7, color=SEG_COLORS['VIP'],
                            xytext=(4,3), textcoords='offset points')
            ax.set_xlabel(x_lab, fontsize=9, color=MUTED)
            xv = x.values; yv = y.values
            mask = np.isfinite(xv) & np.isfinite(yv)
            z = np.polyfit(xv[mask], yv[mask], 1)
            xr = np.linspace(xv[mask].min(), xv[mask].max(), 200)
            ax.plot(xr, np.poly1d(z)(xr), color=SEG_COLORS['VIP'], linewidth=1.8, alpha=0.6)
            r = np.corrcoef(xv[mask], yv[mask])[0,1]

        ax.set_ylabel(y_lab, fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, color='#2C2C2A', pad=8)
        ax.text(0.97, 0.05, f'r = {r:.2f}', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8.5, color=MUTED,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=BG, edgecolor=GRID))
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
        ax.set_facecolor(BG)
        ax.grid(color=GRID, linewidth=0.6, linestyle='--'); ax.set_axisbelow(True)
        ax.tick_params(labelcolor=MUTED, labelsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig
