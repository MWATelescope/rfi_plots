"""Command line interface for rfi_plots."""

import argparse
import sys
from pathlib import Path

from mwalib import CorrelatorContext

from rfi_plots.constants import (
    DEFAULT_COLOUR_MAP,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    LOG_SCALE_MINIMUM_FALLBACK,
    LOG_ZERO_MINIMUM_WARNING,
    MODE_AUTO_AMPLITUDE,
    MODE_DESCRIPTIONS,
    MODE_FLAG_OCCUPANCY,
    MODES,
    OUTPUT_FILENAME_TEMPLATE,
)
from rfi_plots.layout import flagged_tile_report, tile_layout_from_metafits
from rfi_plots.metrics import (
    TileMetric,
    amplitude_statistics,
    clip_to_range,
    compute_auto_amplitudes,
    compute_flag_occupancy,
)
from rfi_plots.plotting import plot_tile_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].

    Returns:
        The parsed arguments.
    """
    mode_help = "; ".join(f"{mode}: {MODE_DESCRIPTIONS[mode]}" for mode in MODES)

    parser = argparse.ArgumentParser(
        prog="rfi-plots",
        description="Plot a to-scale MWA tile map coloured by a per-tile RFI proxy.",
    )
    parser.add_argument(
        "-m",
        "--metafits",
        required=True,
        type=Path,
        help="metafits file for the observation to plot",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=int,
        choices=MODES,
        help=mode_help,
    )
    parser.add_argument(
        "vis_files",
        nargs="+",
        type=Path,
        help="one or more visibility files (gpubox or MWAX channel files)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"output image file (default: {OUTPUT_FILENAME_TEMPLATE})",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="use a logarithmic colour scale (tiles with values of zero or less are drawn as no data)",
    )
    parser.add_argument(
        "--amp-min",
        type=float,
        default=None,
        help="fix the minimum of the amplitude colour scale (default: zero, or the smallest positive "
        "value with --log). Tile values below it are clipped to it and a warning is emitted for each one",
    )
    parser.add_argument(
        "--amp-max",
        type=float,
        default=None,
        help="fix the maximum of the amplitude colour scale (default: taken from the data). "
        "Tile values above it are clipped to it and a warning is emitted for each one",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_COLOUR_MAP,
        help=f"matplotlib colour map name (default: {DEFAULT_COLOUR_MAP})",
    )

    args = parser.parse_args(argv)
    if args.amp_min is not None and args.amp_max is not None and args.amp_min >= args.amp_max:
        parser.error("--amp-min must be less than --amp-max")

    return args


def resolve_amplitude_minimum(amp_min: float | None, log_scale: bool) -> tuple[float | None, str | None]:
    """Resolve the requested amplitude minimum for the chosen colour scale.

    A log scale cannot start at zero, so a requested minimum of zero or less is
    replaced by a small positive value.

    Args:
        amp_min: The requested minimum, or None if it was not given.
        log_scale: True if a log colour scale was requested.

    Returns:
        A tuple of (minimum to use, warning message or None).
    """
    if amp_min is None or not log_scale or amp_min > 0.0:
        return amp_min, None

    return LOG_SCALE_MINIMUM_FALLBACK, LOG_ZERO_MINIMUM_WARNING.format(
        requested=amp_min, used=LOG_SCALE_MINIMUM_FALLBACK
    )


def compute_metric(mode: int, correlator_context: CorrelatorContext) -> TileMetric:
    """Compute the per-tile metric for the requested mode.

    Args:
        mode: Plot mode.
        correlator_context: An open mwalib correlator context.

    Returns:
        The per-tile metric.

    Raises:
        ValueError: The mode is not recognised.
    """
    if mode == MODE_AUTO_AMPLITUDE:
        return compute_auto_amplitudes(correlator_context)

    if mode == MODE_FLAG_OCCUPANCY:
        return compute_flag_occupancy(correlator_context)

    raise ValueError(f"Unrecognised mode: {mode}")


def main(argv: list[str] | None = None) -> int:
    """Run the command line tool.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].

    Returns:
        A process exit code.
    """
    args = parse_args(argv)

    amp_min, amp_min_warning = resolve_amplitude_minimum(args.amp_min, args.log)
    if amp_min_warning is not None:
        print(amp_min_warning, file=sys.stderr)

    try:
        correlator_context = CorrelatorContext(str(args.metafits), [str(path) for path in args.vis_files])
    except Exception as error:  # noqa: BLE001 - mwalib raises many distinct error types
        print(f"Could not open the observation: {error}", file=sys.stderr)
        return EXIT_FAILURE

    metafits_context = correlator_context.metafits_context
    layout = tile_layout_from_metafits(metafits_context)

    for line in flagged_tile_report(layout):
        print(line)

    try:
        metric = compute_metric(args.mode, correlator_context)
    except NotImplementedError as error:
        print(str(error), file=sys.stderr)
        return EXIT_FAILURE

    statistics = amplitude_statistics(metric, layout)

    if amp_min is not None or args.amp_max is not None:
        metric, clip_warnings = clip_to_range(metric, layout, minimum=amp_min, maximum=args.amp_max)
        for warning in clip_warnings:
            print(warning, file=sys.stderr)

    if metric.reads_used == 0:
        print("No visibility data was read from the supplied files.", file=sys.stderr)
        return EXIT_FAILURE

    output_path = args.output or Path(OUTPUT_FILENAME_TEMPLATE.format(obs_id=metafits_context.obs_id, mode=args.mode))
    plot_tile_map(
        layout=layout,
        metric=metric,
        mode=args.mode,
        obs_id=metafits_context.obs_id,
        output_path=output_path,
        colour_map=args.cmap,
        log_scale=args.log,
        value_min=amp_min,
        value_max=args.amp_max,
        statistics=statistics,
    )

    if metric.reads_missing:
        print(f"Skipped {metric.reads_missing} timestep/coarse channel pairs with no data.", file=sys.stderr)

    print(f"Wrote {output_path}")

    return EXIT_SUCCESS
