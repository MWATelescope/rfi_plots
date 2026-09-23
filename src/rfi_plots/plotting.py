"""Tile map plotting."""

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import LogNorm, Normalize

from rfi_plots.constants import (
    AXIS_LABEL_EAST,
    AXIS_LABEL_NORTH,
    COLOUR_BAR_LABELS,
    COLOUR_SCALE_MINIMUM,
    DEFAULT_COLOUR_MAP,
    FIGURE_DPI,
    FIGURE_SIZE_INCHES,
    FLAGGED_TILE_COLOUR,
    GRID_ALPHA,
    LOG_SCALE_LABEL_SUFFIX,
    MATPLOTLIB_BACKEND,
    MINIMUM_POSITIVE_VALUES_FOR_LOG_SCALE,
    MISSING_TILE_COLOUR,
    MODE_DESCRIPTIONS,
    PLOT_MARGIN_FRACTION,
    STATISTICS_SUBTITLE_TEMPLATE,
    TILE_EDGE_COLOUR,
    TILE_EDGE_WIDTH,
    TILE_MARKER,
    TILE_MARKER_SIZE_POINTS2,
)
from rfi_plots.layout import TileLayout
from rfi_plots.metrics import AmplitudeStatistics, TileMetric

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
    value_min: float | None,
    value_max: float | None,
) -> tuple[Normalize | None, bool]:
    """Build the colour normalisation for the plotted values.

    With no limits given a linear scale starts at zero, while a log scale starts
    at the smallest positive value because it cannot start at zero. A log scale
    is refused altogether if every value is zero or negative.

    Args:
        values: The finite per-tile values being plotted.
        log_scale: True if a log colour scale was requested.
        value_min: Lower limit of the colour scale, or None for the default.
            Values of zero or less are ignored on a log scale.
        value_max: Upper limit of the colour scale, or None to take it from the
            values themselves.

    Returns:
        A tuple of (normalisation or None for the matplotlib default, True if a
        log scale is actually in use).
    """
    if not log_scale:
        lower = COLOUR_SCALE_MINIMUM if value_min is None else value_min
        upper = float(np.max(values)) if value_max is None else value_max

        return Normalize(vmin=lower, vmax=upper), False

    positive = values[values > 0.0]
    if positive.size < MINIMUM_POSITIVE_VALUES_FOR_LOG_SCALE:
        return None, False

    lower = float(np.min(positive)) if value_min is None or value_min <= 0.0 else value_min
    upper = float(np.max(positive)) if value_max is None else value_max

    return LogNorm(vmin=lower, vmax=upper), True


def plot_tile_map(
    layout: TileLayout,
    metric: TileMetric,
    mode: int,
    obs_id: int,
    output_path: Path,
    colour_map: str = DEFAULT_COLOUR_MAP,
    log_scale: bool = False,
    value_min: float | None = None,
    value_max: float | None = None,
    statistics: AmplitudeStatistics | None = None,
) -> Path:
    """Plot a to-scale tile map coloured by a per-tile metric.

    Each tile is drawn as a solid square at its east/north position. Tiles
    flagged in the metafits are drawn in a fixed colour rather than by
    amplitude, because their amplitudes are meaningless, and they are left out
    of the colour scale. Unflagged tiles with no data get their own colour.

    Args:
        layout: Tile positions and identities.
        metric: Per-tile values used for the colour scale.
        mode: Plot mode, used for the title and colour bar label.
        obs_id: Observation id, used for the title.
        output_path: File the figure is written to.
        colour_map: Name of the matplotlib colour map to use.
        log_scale: Use a logarithmic colour scale. Tiles with values of zero or
            less are then drawn as no data, because a log scale cannot show them.
        value_min: Lower limit of the colour scale. Defaults to zero on a linear
            scale and to the smallest positive value on a log scale.
        value_max: Upper limit of the colour scale. Defaults to the largest
            plotted value.
        statistics: Statistics over the unclipped values, added to the subtitle
            when given.

    Returns:
        The path the figure was written to.
    """
    has_data = np.isfinite(metric.values) & ~layout.flagged
    norm, log_scale_used = _colour_norm(metric.values[has_data], log_scale, value_min, value_max)

    if log_scale_used:
        has_data &= metric.values > 0.0

    figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    missing = ~has_data & ~layout.flagged
    if np.any(missing):
        axes.scatter(
            layout.east_m[missing],
            layout.north_m[missing],
            s=TILE_MARKER_SIZE_POINTS2,
            marker=TILE_MARKER,
            c=MISSING_TILE_COLOUR,
            edgecolors=TILE_EDGE_COLOUR,
            linewidths=TILE_EDGE_WIDTH,
            label="no data",
        )

    if np.any(layout.flagged):
        axes.scatter(
            layout.east_m[layout.flagged],
            layout.north_m[layout.flagged],
            s=TILE_MARKER_SIZE_POINTS2,
            marker=TILE_MARKER,
            c=FLAGGED_TILE_COLOUR,
            edgecolors=TILE_EDGE_COLOUR,
            linewidths=TILE_EDGE_WIDTH,
            label="flagged",
        )

    tiles = axes.scatter(
        layout.east_m[has_data],
        layout.north_m[has_data],
        s=TILE_MARKER_SIZE_POINTS2,
        marker=TILE_MARKER,
        c=metric.values[has_data],
        cmap=colour_map,
        norm=norm,
        edgecolors=TILE_EDGE_COLOUR,
        linewidths=TILE_EDGE_WIDTH,
    )

    colour_bar = figure.colorbar(tiles, ax=axes)
    colour_bar.set_label(COLOUR_BAR_LABELS[mode] + (LOG_SCALE_LABEL_SUFFIX if log_scale_used else ""))

    axes.set_aspect("equal")
    axes.set_xlim(*_axis_limits(layout.east_m))
    axes.set_ylim(*_axis_limits(layout.north_m))
    axes.set_xlabel(AXIS_LABEL_EAST)
    axes.set_ylabel(AXIS_LABEL_NORTH)
    axes.grid(visible=True, alpha=GRID_ALPHA)
    subtitle = f"{layout.num_tiles} tiles, {metric.reads_used} reads, solid red = flagged in metafits"
    if statistics is not None:
        subtitle += "\n" + STATISTICS_SUBTITLE_TEMPLATE.format(
            mean=statistics.mean,
            standard_deviation=statistics.standard_deviation,
            tiles_used=statistics.tiles_used,
            flagged_excluded=statistics.flagged_excluded,
            zero_excluded=statistics.zero_excluded,
        )

    axes.set_title(f"{obs_id}: mode {mode} - {MODE_DESCRIPTIONS[mode]}\n{subtitle}")

    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return output_path
