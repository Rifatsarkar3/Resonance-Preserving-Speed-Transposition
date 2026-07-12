"""Publication house style for all manuscript figures (single source of truth).

Import and call `apply()` at the top of every figure script, use the semantic colour
roles in PALETTE, tidy axes with `clean(ax)`, and write outputs with `save(fig, name)`
so every figure shares one look and exports vector PDF + >=300-DPI PNG.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FIG_DIR = Path("paper/figures")

# Okabe-Ito colourblind-safe palette, mapped to fixed semantic roles so a colour means
# the same thing in every figure.
PALETTE = {
    "baseline": "#7f7f7f",   # no augmentation / reference (grey)
    "rpst":     "#d55e00",   # RPST (vermillion)
    "resample": "#0072b2",   # resampling / SRA (blue)
    "interp":   "#e69f00",   # RPST-interp / auxiliary (orange)
    "onr":      "#cc79a7",   # order tracking (reddish purple)
    "accent":   "#009e73",   # bluish green
    "true":     "#000000",   # ground-truth reference line
}
# stable categorical map (e.g. fault classes)
CATEGORICAL = ["#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9"]


def apply():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#b0b0b0",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.45,
        "axes.axisbelow": True,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.5,
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def clean(ax):
    """Remove top/right spines for a lighter look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def save(fig, name, width_in=6.5):
    """Write vector PDF + PNG; embed DPI so the PNG sits at `width_in` inches (>=300 DPI)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png, dpi=600)
    plt.close(fig)
    im = Image.open(png); w = im.size[0]
    im.save(png, dpi=(round(w / width_in), round(w / width_in)))
    return png
