"""Base data types for the hybrid_jp."""
from typing import NamedTuple, Type

import numpy as np


# Simulation grid x,y
class Grid(NamedTuple):
    """Simulation grid.

    Note:
        Can be either edges or midpoints.

    Parameters:
        x (np.ndarray): x grid.
        y (np.ndarray): y grid.
    """

    x: np.ndarray
    y: np.ndarray


# Magnetic field components bx, by, bz
class Mag(NamedTuple):
    """Magnetic field components.

    Parameters:
        bx (np.ndarray): x component of magnetic field.
        by (np.ndarray): y component of magnetic field.
        bz (np.ndarray): z component of magnetic field.
    """

    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray

    def is_2d(self) -> bool:
        """Check if all components are 2D arrays.

        Returns:
            bool: True if all components are 2D arrays.
        """
        return all([len(b.shape) == 2 for b in (self.bx, self.by, self.bz)])

    def mean_over_y(self) -> "Mag":
        """Take the mean over the y axis.

        Args:
            mag (Mag): Magnetic field components.

        Returns:
            Mag: Magnetic field components with the mean taken over the y axis.

        Raises:
            ValueError: All components must be 2D arrays.
        """
        if not self.is_2d():
            raise ValueError("All components must be 2D arrays.")
        return Mag(
            bx=np.mean(self.bx, axis=1),
            by=np.mean(self.by, axis=1),
            bz=np.mean(self.bz, axis=1),
        )
