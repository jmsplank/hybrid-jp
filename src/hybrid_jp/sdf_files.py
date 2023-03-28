"""Functions that interact with sdf_helper objects."""
from typing import cast

import numpy as np

from .constants import VAR_BX, VAR_BY, VAR_BZ, VAR_GRID, VAR_GRID_MID
from .dtypes import Grid, Mag


def list_variables(
    data, show_name: bool = True, show_type: bool = False, show_size: bool = True
) -> list[list[str]]:
    """List variables in an sdf_helper object.

    Args:
        data (sdf_helper): sdf_helper object to list variables of.
        show_name (bool, optional): Show variable names. Defaults to True.
        show_type (bool, optional): Show variable types. Defaults to False.
        show_size (bool, optional): Show variable sizes. Defaults to True.

    Returns:
        list[list[str]]: List of variables, where each variable is a list of
            [name, type, size].

    Example:
        >>> list_variables(data)
        [['Grid_Grid', numpy.ndarray, '256, 256'],
        ['Grid_Grid_mid', numpy.ndarray, '256, 256'],
        ['Derived_Number_Density', numpy.ndarray, '256, 256'],
        ['Magnetic_Field_Bx', numpy.ndarray, '256, 256'],
        ['Magnetic_Field_By', numpy.ndarray, '256, 256'],
        ['Magnetic_Field_Bz', numpy.ndarray, '256, 256']]
    """
    dct = data.__dict__
    out = []
    for key in sorted(dct):
        try:
            val = dct[key]
            params = []
            if show_name:
                params.append(key)
            if show_type:
                params.append(type(val))
            if show_size:
                params.append(np.array2string(np.array(val.dims), separator=", "))
            out.append(params)

        except Exception:
            pass
    return out


def print_variables(
    data, show_name: bool = True, show_type: bool = False, show_size: bool = True
) -> None:
    """Print variables in an sdf_helper object.

    Args:
        data (sdf_helper): sdf_helper object to list variables of.
        show_name (bool, optional): Show variable names. Defaults to True.
        show_type (bool, optional): Show variable types. Defaults to False.
        show_size (bool, optional): Show variable sizes. Defaults to True.
    """
    vars = list_variables(data, show_name, show_type, show_size)
    for v in vars:
        print(" ".join(v))


def get_grid(data, mid: bool = False) -> Grid:
    """Get the grid of an sdf_helper object.

    Args:
        data (sdf_helper): sdf data object to get grid of.
        mid (bool, optional): Get the midpoints grid. Defaults to False.

    Returns:
        Grid: Grid namedtuple of x and y grid values.

    Example:
        >>> get_grid(data)
        Grid(x=array([0.00000000e+00, 2.50000000e-05, 5.00000000e-05, ...,
                5.00000000e-02, 5.00250000e-02, 5.00500000e-02]),
            y=array([0.00000000e+00, 2.50000000e-05, 5.00000000e-05, ...,
                5.00000000e-02, 5.00250000e-02, 5.00500000e-02]))
    """
    attr_name = VAR_GRID if not mid else VAR_GRID_MID
    attr = getattr(data, attr_name).data
    return Grid(x=attr[0], y=attr[1])


def get_mag(data) -> Mag:
    # google style docstring
    """Get the magnetic field of an sdf_helper object.

    Args:
        data (sdf_helper): sdf data object to get magnetic field of.

    Returns:
        Mag: Mag namedtuple of x, y, and z magnetic field values.

    Example:
        >>> get_mag(data)
        Mag(x=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,]],
            y=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]],
            z=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]]))
    """
    components = []
    for i in [VAR_BX, VAR_BY, VAR_BZ]:
        components.append(getattr(data, i).data)
    return Mag._make(components)
