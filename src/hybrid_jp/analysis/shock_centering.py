import numpy as np
from epoch_cheats.deck import Deck

from hybrid_jp.sdf_files import SDF


class CenteredShock:
    def __init__(self, sdfs: list[SDF], deck: Deck) -> None:
        self.sdfs = sdfs
        self.deck = deck

        self.grid_km = self.sdfs[0].mid_grid * (1 / 1000)
        self._nd_median_y_cm = np.asarray(
            [np.median(sdf.numberdensity, axis=1) for sdf in sdfs]
        ).T / (100**3)

        self.shock_i = self.shock_index_from_nd()
        self.shock_x = self.grid_km.x[self.shock_i]

    def shock_index_from_nd(self, threshold=0.37):
        mask = self._nd_median_y_cm > (self._nd_median_y_cm.max() * threshold)
        shock_i = np.asarray(np.argmax(mask, axis=0))
        return shock_i
