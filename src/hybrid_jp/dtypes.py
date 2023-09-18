"""Base data types for the hybrid_jp."""
from dataclasses import dataclass
from typing import Iterator, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

BaseChild = TypeVar("BaseChild", bound="BaseContainer")
arrfloat = npt.NDArray[np.float64]
arrint = npt.NDArray[np.int_]


@dataclass
class BaseContainer(Protocol):
    """Base container for parameters."""

    @property
    def all(self) -> dict[str, arrfloat]:
        """Return all parameters as a dict."""
        ...

    def __mul__(self: BaseChild, value: float | int) -> BaseChild:
        """Multiply."""
        return type(self)(**{k: v * value for k, v in self.all.items()})

    def __rmul__(self: BaseChild, value: float | int) -> BaseChild:
        """Multiply."""
        return self.__mul__(value)

    def __imul__(self: BaseChild, value: float | int) -> BaseChild:
        """Multiply."""
        return self.__mul__(value)

    def __iter__(self) -> Iterator[arrfloat]:
        """Iterate over all values."""
        for v in self.all.values():
            yield v

    def is_2d(self) -> bool:
        """Check if all components are 2D arrays.

        Returns:
            bool: True if all components are 2D arrays.
        """
        return all([len(b.shape) == 2 for b in self.all.values()])

    def mean_over_axis(self: BaseChild, axis: int) -> BaseChild:
        """Take the mean over the specified axis in a 2d array.

        axis=0: mean over x
        axis=1: mean over y

        Args:
            axis (int): Axis to take the mean over.

        Raises:
            ValueError: All components must be 2D arrays.
        """
        if not self.is_2d():
            raise ValueError("All components must be 2D arrays.")
        return type(self)(**{k: np.mean(v, axis=axis) for k, v in self.all.items()})

    def mean_over_y(self: BaseChild) -> BaseChild:
        """Take the mean over the y axis.

        Returns:
            Mag: Magnetic field components with the mean taken over the y axis.

        Raises:
            ValueError: All components must be 2D arrays.
        """
        return self.mean_over_axis(axis=1)

    def slice_x(self: BaseChild, start: int, stop: int) -> BaseChild:
        """Slice in the x direction."""
        if self.is_2d():
            return type(self)(**{k: v[start:stop, :] for k, v in self.all.items()})
        else:
            items = list(self.all.items())
            items[0] = (items[0][0], items[0][1][start:stop])
            return type(self)(**dict(items))


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
    def all(self) -> dict[str, arrfloat]:
        """All parameters as a dict."""
        return dict(x=self.x, y=self.y)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the grid (nx, ny)."""
        return self.x.shape[0], self.y.shape[0]


@dataclass
class Mag(BaseContainer):
    """Magnetic field components.

    Parameters:
        bx (np.ndarray): x component of magnetic field.
        by (np.ndarray): y component of magnetic field.
        bz (np.ndarray): z component of magnetic field.
    """

    bx: np.ndarray
    by: np.ndarray
    bz: np.ndarray

    @property
    def all(self) -> dict[str, arrfloat]:
        """All parameters as a dict."""
        return dict(bx=self.bx, by=self.by, bz=self.bz)


@dataclass
class Elec(BaseContainer):
    """Electric field components.

    Parameters:
        ex (np.ndarray): x component of electric field.
        ey (np.ndarray): y component of electric field.
        ez (np.ndarray): z component of electric field.
    """

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray

    @property
    def all(self) -> dict[str, arrfloat]:
        """All parameters as a dict."""
        return dict(ex=self.ex, ey=self.ey, ez=self.ez)


@dataclass
class Current(BaseContainer):
    """Current components."""

    jx: arrfloat
    jy: arrfloat
    jz: arrfloat

    @property
    def all(self) -> dict[str, arrfloat]:
        """All currents.

        Returns:
            dict[str, arrf]: 'jx'|'jy'|'jz', (x, y)
        """
        return dict(jx=self.jx, jy=self.jy, jz=self.jz)
