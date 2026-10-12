"""Shared, local styling for moeforecast diagnostic plots."""

from __future__ import annotations

PALETTE = (
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
)


def style_axis(ax, grid_axis: str = "both"):
    """Apply the serif, light-grid style used throughout the reference notebooks."""
    ax.set_facecolor("white")
    ax.grid(
        True,
        axis=grid_axis,
        color="#cbd5df",
        alpha=0.55,
        linewidth=0.65,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aab7c4")
    ax.spines["bottom"].set_color("#aab7c4")
    ax.tick_params(colors="#34495e")
    ax.xaxis.label.set_fontfamily("serif")
    ax.yaxis.label.set_fontfamily("serif")
    ax.title.set_fontfamily("serif")
