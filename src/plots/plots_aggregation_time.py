import matplotlib.pyplot as plt
import textwrap
from src.utils.style import apply_style
import os
from typing import Union
import matplotlib


COLORS = apply_style()
BG = COLORS.get("bg", "#FAFAF9")
AGGREGATION_TYPE = "time"


def plot_distributions(
    df_clean,
    output_path: Union[str, os.PathLike] = f"distributions_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    METRICS = ["users","visits","clicks","click_depth",
               "dwell_time_avg","return_rate","dwelltime_l"]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(
        "Distribución global de ratios fin-semana/(fin+entre-semana)\nReferencia: 0.5 = igual comportamiento",
        fontsize=14, fontweight="bold", color=COLORS["title"]
    )
    axes = axes.flatten()

    for i, m in enumerate(METRICS):
        ax = axes[i]
        data = df_clean[m].dropna()

        ax.hist(
            data, bins=25,
            color=COLORS["bars"]["view"],
            edgecolor="white", linewidth=0.5, alpha=0.85
        )

        ax.axvline(0.5, color=COLORS["lines"]["ratio_cart_purchase"], lw=2, ls="--")
        ax.axvline(data.mean(), color=COLORS["lines"]["ratio_view_cart"], lw=2)

        ax.set_title(m, fontsize=11, fontweight="bold", color=COLORS["title"])
        ax.set_xlabel("ratio")
        ax.set_ylabel("frecuencia")

    axes[-1].set_visible(False)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_descriptive_table(
    df_clean,
    output_path: Union[str, os.PathLike] = f"descriptive_table_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    METRICS = ["users","visits","clicks","click_depth",
               "dwell_time_avg","return_rate","dwelltime_l"]

    desc = df_clean[METRICS].describe().T.round(4)
    desc["bias"] = (desc["mean"] - 0.5).round(4)
    desc.index = [m.replace("_"," ") for m in desc.index]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    table = ax.table(
        cellText=desc.values,
        rowLabels=desc.index,
        colLabels=["count","mean","std","min","25%","50%","75%","max","bias"],
        cellLoc="center", rowLoc="center", loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    for r in range(1, len(desc)+1):
        bias_val = desc["bias"].iloc[r-1]
        color = "#d4edda" if bias_val > 0 else "#f8d7da"
        table[r, 8].set_facecolor(color)

    for c in range(9):
        table[0, c].set_facecolor(COLORS["title"])
        table[0, c].set_text_props(color="white", fontweight="bold")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_heatmap_root(
    df_clean,
    output_path: Union[str, os.PathLike] = f"heatmap_root_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    METRICS = ["users","visits","clicks","click_depth",
               "dwell_time_avg","return_rate","dwelltime_l"]

    heat = df_clean.groupby("root")[METRICS].mean().round(4)
    heat_bias = heat - 0.5

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(heat_bias.values, cmap="RdYlGn", vmin=-0.25, vmax=0.25, aspect="auto")

    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([m.replace("_","\n") for m in METRICS])
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index)

    ax.set_title(
        "Bias relativo al 0.5 por categoría raíz y métrica",
        fontsize=13, fontweight="bold", color=COLORS["title"]
    )

    for i in range(len(heat)):
        for j in range(len(METRICS)):
            val = heat_bias.values[i, j]
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(val) < 0.15 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_scatter_users_dwell(
    df_clean,
    output_path: Union[str, os.PathLike] = f"scatter_users_dwell_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    roots = df_clean["root"].unique()
    cmap4 = plt.get_cmap("tab20", len(roots))
    color_map = {r: cmap4(i) for i, r in enumerate(roots)}

    fig, ax = plt.subplots(figsize=(11, 7))

    for root, grp in df_clean.groupby("root"):
        ax.scatter(
            grp["users"], grp["dwell_time_avg"],
            color=color_map[root], label=root,
            s=60, alpha=0.8, edgecolors="white", lw=0.4
        )

    ax.axhline(0.5, color=COLORS["lines"]["ratio_cart_purchase"], lw=2, ls="--")
    ax.axvline(0.5, color=COLORS["lines"]["ratio_cart_purchase"], lw=2, ls="--")

    ax.set_xlabel("ratio users")
    ax.set_ylabel("ratio dwell_time_avg")
    ax.set_title("Ratio users vs dwell_time_avg", fontsize=13, fontweight="bold", color=COLORS["title"])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_top15_dwell(
    df_clean,
    output_path: Union[str, os.PathLike] = f"top15_dwell_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    top15_high = df_clean.nlargest(15, "dwell_time_avg")[["category_code","dwell_time_avg"]]
    top15_low  = df_clean.nsmallest(15, "dwell_time_avg")[["category_code","dwell_time_avg"]]

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(17, 6))

    def barh_plot(ax, data, col, title, color):
        labs = [textwrap.shorten(c, width=30, placeholder="…") for c in data["category_code"]]
        vals = data[col].values
        bars = ax.barh(labs, vals, color=color, edgecolor="white", height=0.65)

        ax.axvline(0.5, color=COLORS["lines"]["ratio_cart_purchase"], lw=2, ls="--")

        for bar, v in zip(bars, vals):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=12, fontweight="bold", color=COLORS["title"])
        ax.invert_yaxis()

    barh_plot(ax_h, top15_high, "dwell_time_avg",
              "Top 15 — Mayor ratio dwell_time_avg", COLORS["bars"]["cart"])

    barh_plot(ax_l, top15_low, "dwell_time_avg",
              "Top 15 — Menor ratio dwell_time_avg", COLORS["bars"]["purchase"])

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_correlation_matrix(
    df_clean,
    output_path: Union[str, os.PathLike] = f"correlation_matrix_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    METRICS = ["users","visits","clicks","click_depth",
               "dwell_time_avg","return_rate","dwelltime_l"]

    corr = df_clean[METRICS].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    labels = [m.replace("_","\n") for m in METRICS]
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(labels)

    ax.set_title("Correlación entre ratios de métricas",
                 fontsize=13, fontweight="bold", color=COLORS["title"])

    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(corr.values[i,j]) > 0.6 else "black")

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig


def plot_boxplots_roots(
    df_clean,
    output_path: Union[str, os.PathLike] = f"boxplots_roots_{AGGREGATION_TYPE}.png",
    COLORS=COLORS
):
    ROOT_ORDER = sorted(df_clean["root"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, color in [
        (ax1, "dwell_time_avg", COLORS["bars"]["view"]),
        (ax2, "clicks", COLORS["bars"]["cart"])
    ]:

        groups = [df_clean[df_clean["root"]==r][metric].values for r in ROOT_ORDER]
        bp = ax.boxplot(groups, patch_artist=True, vert=True,
                        medianprops=dict(color="white", lw=2))

        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(ROOT_ORDER)+1))
        ax.set_xticklabels(ROOT_ORDER, rotation=40, ha="right")

        ax.axhline(0.5, color=COLORS["lines"]["ratio_cart_purchase"], lw=2, ls="--")

        ax.set_title(f"Distribución de ratio {metric}",
                     fontsize=12, fontweight="bold", color=COLORS["title"])
        ax.set_ylabel("ratio")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    return fig
