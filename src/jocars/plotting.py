"""One shared chart style (validated reference palette from the dataviz skill)."""
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order: blue, orange, aqua (first three pass all-pairs CVD checks)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = "#2a78d6"  # single-hue sequential / single-series color


def set_style() -> None:
    """Apply a clean, recessive style: thin grid, no top/right spines, readable text."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "text.color": TEXT,
        "font.size": 10.5, "legend.frameon": False, "lines.linewidth": 2,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def jod(x, _pos=None) -> str:
    """Axis formatter: 12500 -> '12.5k'."""
    return f"{x/1000:g}k" if x >= 1000 else f"{x:.0f}"
