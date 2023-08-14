"""Base data types for the hybrid_jp."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass
class BaseContainer(Protocol):
    """Base container for parameters."""

    @property
    def all(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return all parameters as a dict."""
        ...


@dataclass
class Grid(BaseContainer):
    """Simulation grid.

    Note:
        Can be either edges or midpoints.

    Parameters:
        x (np.ndarray): x grid.
        y (np.ndarray): y grid.
    """

    x: np.ndarray
    y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the grid (nx, ny)."""
        return self.x.shape[0], self.y.shape[0]

    def __mul__(self, value: float | int) -> "Grid":
        """Multiply."""
        return Grid(x=self.x * value, y=self.y * value)

    def __rmul__(self, value: float | int) -> "Grid":
        """Multiply."""
        return self.__mul__(value)

    def __imul__(self, value: float | int) -> "Grid":
        """Multiply."""
        return self.__mul__(value)

    def __iter__(self):
        """Iterate over x and y."""
        yield self.x
        yield self.y

    def slice_x(self, start: int, stop: int) -> "Grid":
        """Slice in the x direction.

        Args:
            start (int): Start index.
            stop (int): Stop index.

        Returns:
            Grid: Sliced grid Gird(x=x[start:stop], y=y).
        """
        return Grid(x=self.x[start:stop].copy(), y=self.y.copy())


@dataclass
class Mag:
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

    def slice_x(self, start: int, stop: int) -> "Mag":
        """Slice the data along x.

        Args:
            start (int): Start index.
            stop (int): Stop index.

        Returns:
            Mag: Sliced magnetic field.
        """
        return Mag(
            bx=self.bx[start:stop, :],
            by=self.by[start:stop, :],
            bz=self.bz[start:stop, :],
        )
