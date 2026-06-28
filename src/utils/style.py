import matplotlib.pyplot as plt
import matplotlib as mpl


def apply_style():
    """
    Aplica el estilo visual estandarizado utilizado en el TFM de Engagement en E-commerce.
    Este estilo replica exactamente la estética del gráfico base proporcionado.
    """

    # --- Paleta de colores ---
    COLORS = {
        "background": "#FAFAF9",
        "text": "#5F5E5A",
        "title": "#2C2C2A",
        "grid": "#D3D1C7",
        "bars": {
            "view": "#378ADD",
            "cart": "#1D9E75",
            "purchase": "#D85A30",
        },
        "lines": {
            "ratio_view_cart": "#7F77DD",
            "ratio_cart_purchase": "#D4537E",
        }
    }

    # --- Estilo global de Matplotlib ---
    mpl.rcParams.update({

        # Fondo general
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": COLORS["background"],

        # Texto
        "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],

        # Tipografía
        "font.size": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,

        # Grid
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Líneas
        "lines.linewidth": 2.2,

        # Leyenda
        "legend.framealpha": 0.6,
        "legend.facecolor": COLORS["background"],
        "legend.edgecolor": COLORS["grid"],

        # Tamaño por defecto de figura
        "figure.figsize": (11, 6),

        # Ajuste automático
        "figure.autolayout": True,
    })

    return COLORS
