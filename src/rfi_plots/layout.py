"""Tile layout taken from an MWA metafits file via mwalib."""

from dataclasses import dataclass

import numpy as np
from mwalib import MetafitsContext
from numpy.typing import NDArray


@dataclass(frozen=True)
class TileLayout:
    """Physical positions and identities of the tiles in an observation.

    All arrays are indexed by mwalib antenna index and have the same length.

    Attributes:
        tile_ids: Tile id of each antenna, as recorded in the metafits.
        tile_names: Tile name of each antenna, for example ``Tile051``.
        east_m: Metres east of the array centre.
        north_m: Metres north of the array centre.
        flagged: True where either the X or the Y RF input is flagged.
    """

    tile_ids: NDArray[np.int64]
    tile_names: list[str]
    east_m: NDArray[np.float64]
    north_m: NDArray[np.float64]
    flagged: NDArray[np.bool_]

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles in the layout."""
        return len(self.tile_names)


def tile_layout_from_metafits(metafits_context: MetafitsContext) -> TileLayout:
    """Build a TileLayout from a metafits context.

    Args:
        metafits_context: An open mwalib metafits context.

    Returns:
        The tile layout for the observation described by the metafits.
    """
    antennas = metafits_context.antennas

    return TileLayout(
        tile_ids=np.array([antenna.tile_id for antenna in antennas], dtype=np.int64),
        tile_names=[antenna.tile_name for antenna in antennas],
        east_m=np.array([antenna.east_m for antenna in antennas], dtype=np.float64),
        north_m=np.array([antenna.north_m for antenna in antennas], dtype=np.float64),
        flagged=np.array(
            [antenna.rfinput_x.flagged or antenna.rfinput_y.flagged for antenna in antennas],
            dtype=np.bool_,
        ),
    )
