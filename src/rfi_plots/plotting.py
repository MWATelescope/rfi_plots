"""Tile map plotting."""

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from rfi_plots.constants import (
    AXIS_LABEL_EAST,
    AXIS_LABEL_NORTH,
    COLOUR_BAR_LABELS,
    DEFAULT_COLOUR_MAP,
    FIGURE_DPI,
    FIGURE_SIZE_INCHES,
    FLAGGED_TILE_EDGE_COLOUR,
    FLAGGED_TILE_EDGE_WIDTH,
    GRID_ALPHA,
    LOG_SCALE_LABEL_SUFFIX,
    MATPLOTLIB_BACKEND,
    MINIMUM_POSITIVE_VALUES_FOR_LOG_SCALE,
    MISSING_TILE_COLOUR,
    MODE_DESCRIPTIONS,
    PLOT_MARGIN_FRACTION,
    TILE_EDGE_COLOUR,
    TILE_EDGE_WIDTH,
    TILE_MARKER,
    TILE_MARKER_SIZE_POINTS2,
)
from rfi_plots.layout import TileLayout
from rfi_plots.metrics import TileMetric

matplotlib.use(MATPLOTLIB_BACKEND)

import matplotlib.pyplot as plt  # noqa: E402


def _axis_limits(positions: np.ndarray) -> tuple[float, float]:
    """Return axis limits with a fractional margin around the tile positions.

    Args:
        positions: Tile positions along one axis, in metres.

    Returns:
        A tuple of (lower limit, upper limit) in metres.
    """
    lower = float(np.min(positions))
    upper = float(np.max(positions))
    margin = (upper - lower) * PLOT_MARGIN_FRACTION

    return lower - margin, upper + margin


def _colour_norm(
    values: np.ndarray,
    log_scale: bool,
    value_max: float | None,
) -> tuple[Normalize | None, bool]:
    """Build the colour normalisation for the plotted values.

    A log scale needs at least one positive value, so it is refused if every
    value is zero or negative.

    Args:
        values: The finite per-tile values being plotted.
        log_scale: True if a log colour scale was requested.
        value_max: Upper limit of the colour scale, or None to take it from the
            values themselves.

    Returns:
        A tuple of (normalisation or None for the matplotlib default, True if a
        log scale is actually in use).
    """
    if not log_scale:
        if value_max is None:
            return None, False

        return Normalize(vmin=float(np.min(values)), vmax=value_max), False

    positive = values[values > 0.0]
    if positive.size < MINIMUM_POSITIVE_VALUES_FOR_LOG_SCALE:
        return None, False

    upper = float(np.max(positive)) if value_max is None else value_max

    return LogNorm(vmin=float(np.min(positive)), vmax=upper), True


def plot_tile_map(
    layout: TileLayout,
    metric: TileMetric,
    mode: int,
    obs_id: int,
    output_path: Path,
    colour_map: str = DEFAULT_COLOUR_MAP,
    log_scale: bool = False,
    value_max: float | None = None,
) -> Path:
    """Plot a to-scale tile map coloured by a per-tile metric.

    Each tile is drawn as a solid square at its east/north position. Tiles
    flagged in the metafits get a coloured edge, and tiles with no data are
    drawn in a fixed colour.

    Args:
        layout: Tile positions and identities.
        metric: Per-tile values used for the colour scale.
        mode: Plot mode, used for the title and colour bar label.
        obs_id: Observation id, used for the title.
        output_path: File the figure is written to.
        colour_map: Name of the matplotlib colour map to use.
        log_scale: Use a logarithmic colour scale. Tiles with values of zero or
            less are then drawn as no data, because a log scale cannot show them.
        value_max: Upper limit of the colour scale. Defaults to the largest
            plotted value.

    Returns:
        The path the figure was written to.
    """
    has_data = np.isfinite(metric.values)
    norm, log_scale_used = _colour_norm(metric.values[has_data], log_scale, value_max)

    if log_scale_used:
        has_data &= metric.values > 0.0

    figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    edge_colours = np.where(layout.flagged, FLAGGED_TILE_EDGE_COLOUR, TILE_EDGE_COLOUR)
    edge_widths = np.where(layout.flagged, FLAGGED_TILE_EDGE_WIDTH, TILE_EDGE_WIDTH)

    if np.any(~has_data):
        axes.scatter(
            layout.east_m[~has_data],
            layout.north_m[~has_data],
            s=TILE_MARKER_SIZE_POINTS2,
            marker=TILE_MARKER,
            c=MISSING_TILE_COLOUR,
            edgecolors=edge_colours[~has_data],
            linewidths=edge_widths[~has_data],
            label="no data",
        )

    tiles = axes.scatter(
        layout.east_m[has_data],
        layout.north_m[has_data],
        s=TILE_MARKER_SIZE_POINTS2,
        marker=TILE_MARKER,
        c=metric.values[has_data],
        cmap=colour_map,
        norm=norm,
        edgecolors=edge_colours[has_data],
        linewidths=edge_widths[has_data],
    )

    colour_bar = figure.colorbar(tiles, ax=axes)
    colour_bar.set_label(COLOUR_BAR_LABELS[mode] + (LOG_SCALE_LABEL_SUFFIX if log_scale_used else ""))

    axes.set_aspect("equal")
    axes.set_xlim(*_axis_limits(layout.east_m))
    axes.set_ylim(*_axis_limits(layout.north_m))
    axes.set_xlabel(AXIS_LABEL_EAST)
    axes.set_ylabel(AXIS_LABEL_NORTH)
    axes.grid(visible=True, alpha=GRID_ALPHA)
    axes.set_title(
        f"{obs_id}: mode {mode} - {MODE_DESCRIPTIONS[mode]}\n"
        f"{layout.num_tiles} tiles, {metric.reads_used} reads, "
        f"red edge = flagged in metafits"
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return output_path
