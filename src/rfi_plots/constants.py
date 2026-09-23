"""Constants for the rfi_plots package."""

# Plot modes selectable on the command line.
MODE_AUTO_AMPLITUDE = 1
MODE_FLAG_OCCUPANCY = 2
MODES = (MODE_AUTO_AMPLITUDE, MODE_FLAG_OCCUPANCY)

MODE_DESCRIPTIONS = {
    MODE_AUTO_AMPLITUDE: "autocorrelation amplitude from raw visibilities",
    MODE_FLAG_OCCUPANCY: "aoflagger flag occupancy from preprocessed visibilities",
}

# mwalib returns visibilities as [baseline][fine_chan][pol, r, i], which is
# 8 floats per fine channel in XX, XY, YX, YY order.
VIS_XX_REAL_INDEX = 0
VIS_XX_IMAG_INDEX = 1
VIS_YY_REAL_INDEX = 6
VIS_YY_IMAG_INDEX = 7

# A visibility is gain_i * conj(gain_j), so an autocorrelation scales with the
# square of the digital gain of its RF input.
DIGITAL_GAIN_POWER_EXPONENT = 2

# Number of polarisations averaged into the mode 1 metric (XX and YY).
AUTO_POLS_AVERAGED = 2

# Plot appearance.
MATPLOTLIB_BACKEND = "Agg"
FIGURE_SIZE_INCHES = (10.0, 8.5)
FIGURE_DPI = 150
DEFAULT_COLOUR_MAP = "viridis"
TILE_MARKER = "s"
TILE_MARKER_SIZE_POINTS2 = 110.0
TILE_EDGE_COLOUR = "black"
TILE_EDGE_WIDTH = 0.4
FLAGGED_TILE_EDGE_COLOUR = "red"
FLAGGED_TILE_EDGE_WIDTH = 1.6
MISSING_TILE_COLOUR = "lightgrey"
GRID_ALPHA = 0.25
PLOT_MARGIN_FRACTION = 0.05
AXIS_LABEL_EAST = "East of array centre (m)"
AXIS_LABEL_NORTH = "North of array centre (m)"

COLOUR_BAR_LABELS = {
    MODE_AUTO_AMPLITUDE: "Mean autocorrelation amplitude (XX, YY)",
    MODE_FLAG_OCCUPANCY: "Flag occupancy (fraction)",
}

# Output file naming.
OUTPUT_FILENAME_TEMPLATE = "{obs_id}_mode{mode}_tile_map.png"

# Process exit codes.
EXIT_SUCCESS = 0
EXIT_FAILURE = 1

# Log colour scale.
LOG_SCALE_LABEL_SUFFIX = " (log scale)"
MINIMUM_POSITIVE_VALUES_FOR_LOG_SCALE = 1

# The linear amplitude colour scale always starts at zero. A log scale cannot,
# so it starts at the smallest positive plotted value.
COLOUR_SCALE_MINIMUM = 0.0

# A log scale cannot start at zero, so this is used instead when the requested
# minimum is zero or less.
LOG_SCALE_MINIMUM_FALLBACK = 1.0
LOG_ZERO_MINIMUM_WARNING = (
    "--amp-min of {requested:.6g} cannot be used with --log, so a minimum of {used:.6g} was used instead"
)

# Amplitude range overrides.
CLIP_WARNING_TEMPLATE = "tile: {ant},{tile_id},{tile_name} amplitude {amplitude:.6g} has been clipped to {limit:.6g}"

# Flagged tile report.
FLAGGED_TILE_REPORT_HEADER = "{count} tile(s) flagged in the metafits:"
FLAGGED_TILE_TEMPLATE = "tile: {ant},{tile_id},{tile_name}"
NO_FLAGGED_TILES_MESSAGE = "No tiles are flagged in the metafits"
