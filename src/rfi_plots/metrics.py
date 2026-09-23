"""Per-tile metrics used to colour the tile map."""

from dataclasses import dataclass, replace

import numpy as np
from mwalib import CorrelatorContext, GpuboxErrorNoDataForTimeStepCoarseChannel, MetafitsContext
from numpy.typing import NDArray

from rfi_plots.constants import (
    AUTO_POLS_AVERAGED,
    CLIP_WARNING_TEMPLATE,
    DIGITAL_GAIN_POWER_EXPONENT,
    VIS_XX_IMAG_INDEX,
    VIS_XX_REAL_INDEX,
    VIS_YY_IMAG_INDEX,
    VIS_YY_REAL_INDEX,
)
from rfi_plots.layout import TileLayout


@dataclass(frozen=True)
class TileMetric:
    """A per-tile scalar value plus a record of what went into it.

    Attributes:
        values: One value per mwalib antenna index. NaN where no data was read.
        reads_used: Number of (timestep, coarse channel) reads that contributed.
        reads_missing: Number of (timestep, coarse channel) pairs with no data.
        digital_gains_corrected: True if a digital gain correction was divided out.
    """

    values: NDArray[np.float64]
    reads_used: int
    reads_missing: int
    digital_gains_corrected: bool


def _autocorrelation_baseline_map(metafits_context: MetafitsContext) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Find the baseline indices that hold autocorrelations.

    Args:
        metafits_context: An open mwalib metafits context.

    Returns:
        A tuple of (baseline indices, antenna indices) for the autocorrelations.
    """
    pairs = [
        (baseline_index, baseline.ant1_index)
        for baseline_index, baseline in enumerate(metafits_context.baselines)
        if baseline.ant1_index == baseline.ant2_index
    ]
    baseline_indices = np.array([pair[0] for pair in pairs], dtype=np.int64)
    antenna_indices = np.array([pair[1] for pair in pairs], dtype=np.int64)

    return baseline_indices, antenna_indices


def _digital_gain_power(
    metafits_context: MetafitsContext,
    receiver_channel_number: int,
    antenna_indices: NDArray[np.int64],
) -> NDArray[np.float64] | None:
    """Get the per-input digital gain power for one coarse channel.

    Cable and geometric delays are phase only, so they do not change
    autocorrelation amplitudes. The per-input digital gain does, so it is
    divided out here when the correlator has not already done so.

    Args:
        metafits_context: An open mwalib metafits context.
        receiver_channel_number: Receiver coarse channel number being read.
        antenna_indices: Antenna index of each autocorrelation, in read order.

    Returns:
        An array of shape (num_autos, 2) holding the squared X and Y digital
        gains, or None if the correlator already applied the digital gains.
    """
    if metafits_context.digital_gains_applied:
        return None

    gain_index_by_receiver_channel = {
        coarse_chan.rec_chan_number: index for index, coarse_chan in enumerate(metafits_context.metafits_coarse_chans)
    }
    gain_index = gain_index_by_receiver_channel[receiver_channel_number]
    antennas = metafits_context.antennas

    gains = np.array(
        [
            [
                antennas[antenna_index].rfinput_x.digital_gains[gain_index],
                antennas[antenna_index].rfinput_y.digital_gains[gain_index],
            ]
            for antenna_index in antenna_indices
        ],
        dtype=np.float64,
    )

    return gains**DIGITAL_GAIN_POWER_EXPONENT


def compute_auto_amplitudes(correlator_context: CorrelatorContext) -> TileMetric:
    """Compute the mean autocorrelation amplitude per tile (mode 1).

    The XX and YY autocorrelation amplitudes are averaged over every fine
    channel, timestep and coarse channel that the supplied visibility files
    provide.

    Args:
        correlator_context: An open mwalib correlator context.

    Returns:
        The per-tile mean autocorrelation amplitude.
    """
    metafits_context = correlator_context.metafits_context
    baseline_indices, antenna_indices = _autocorrelation_baseline_map(metafits_context)

    totals = np.zeros(metafits_context.num_ants, dtype=np.float64)
    counts = np.zeros(metafits_context.num_ants, dtype=np.int64)
    reads_used = 0
    reads_missing = 0
    gains_corrected = False

    for coarse_chan_index in correlator_context.provided_coarse_chan_indices:
        receiver_channel_number = correlator_context.coarse_chans[coarse_chan_index].rec_chan_number
        gain_power = _digital_gain_power(metafits_context, receiver_channel_number, antenna_indices)
        gains_corrected = gains_corrected or gain_power is not None

        for timestep_index in correlator_context.provided_timestep_indices:
            try:
                data = correlator_context.read_by_baseline(timestep_index, coarse_chan_index)
            except GpuboxErrorNoDataForTimeStepCoarseChannel:
                reads_missing += 1
                continue

            autos = data[baseline_indices].astype(np.float64)
            xx_amplitude = np.hypot(autos[:, :, VIS_XX_REAL_INDEX], autos[:, :, VIS_XX_IMAG_INDEX])
            yy_amplitude = np.hypot(autos[:, :, VIS_YY_REAL_INDEX], autos[:, :, VIS_YY_IMAG_INDEX])

            if gain_power is not None:
                xx_amplitude /= gain_power[:, 0, np.newaxis]
                yy_amplitude /= gain_power[:, 1, np.newaxis]

            mean_amplitude = np.mean(xx_amplitude + yy_amplitude, axis=1) / AUTO_POLS_AVERAGED
            totals[antenna_indices] += mean_amplitude
            counts[antenna_indices] += 1
            reads_used += 1

    values = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)

    return TileMetric(
        values=values,
        reads_used=reads_used,
        reads_missing=reads_missing,
        digital_gains_corrected=gains_corrected,
    )


def compute_flag_occupancy(correlator_context: CorrelatorContext) -> TileMetric:
    """Compute the per-tile aoflagger flag occupancy (mode 2).

    Args:
        correlator_context: An open mwalib correlator context.

    Raises:
        NotImplementedError: Mode 2 is not implemented yet.
    """
    raise NotImplementedError(
        "Mode 2 (flag occupancy) is not implemented yet. It needs Birli/aoflagger output "
        "(uvfits or measurement set) rather than raw visibility files."
    )


def clip_to_maximum(metric: TileMetric, layout: TileLayout, maximum: float) -> tuple[TileMetric, list[str]]:
    """Clip per-tile values to a maximum and report each tile that was clipped.

    Args:
        metric: The per-tile metric to clip.
        layout: Tile identities, used to name the clipped tiles.
        maximum: The value that anything larger is clipped to.

    Returns:
        A tuple of (clipped metric, one warning message per clipped tile).
    """
    clipped_mask = np.isfinite(metric.values) & (metric.values > maximum)
    messages = [
        CLIP_WARNING_TEMPLATE.format(
            ant=antenna_index,
            tile_id=layout.tile_ids[antenna_index],
            tile_name=layout.tile_names[antenna_index],
            amplitude=metric.values[antenna_index],
            maximum=maximum,
        )
        for antenna_index in np.flatnonzero(clipped_mask)
    ]

    return replace(metric, values=np.where(clipped_mask, maximum, metric.values)), messages
