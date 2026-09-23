"""Command line interface for rfi_plots."""

import argparse
import sys
from pathlib import Path

from mwalib import CorrelatorContext

from rfi_plots.constants import (
    DEFAULT_COLOUR_MAP,
    EXIT_FAILURE,
    EXIT_SUCCESS,
    MODE_AUTO_AMPLITUDE,
    MODE_DESCRIPTIONS,
    MODE_FLAG_OCCUPANCY,
    MODES,
    OUTPUT_FILENAME_TEMPLATE,
)
from rfi_plots.layout import tile_layout_from_metafits
from rfi_plots.metrics import TileMetric, compute_auto_amplitudes, compute_flag_occupancy
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
        "--cmap",
        default=DEFAULT_COLOUR_MAP,
        help=f"matplotlib colour map name (default: {DEFAULT_COLOUR_MAP})",
    )

    return parser.parse_args(argv)


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

    try:
        correlator_context = CorrelatorContext(str(args.metafits), [str(path) for path in args.vis_files])
    except Exception as error:  # noqa: BLE001 - mwalib raises many distinct error types
        print(f"Could not open the observation: {error}", file=sys.stderr)
        return EXIT_FAILURE

    metafits_context = correlator_context.metafits_context
    layout = tile_layout_from_metafits(metafits_context)

    try:
        metric = compute_metric(args.mode, correlator_context)
    except NotImplementedError as error:
        print(str(error), file=sys.stderr)
        return EXIT_FAILURE

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
    )

    if metric.reads_missing:
        print(f"Skipped {metric.reads_missing} timestep/coarse channel pairs with no data.", file=sys.stderr)

    print(f"Wrote {output_path}")

    return EXIT_SUCCESS
