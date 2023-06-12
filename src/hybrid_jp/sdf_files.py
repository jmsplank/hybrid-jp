"""Functions that interact with sdf_helper objects."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sdf_helper as sh
from sdf import BlockList

from .dtypes import Grid, Mag


def load(path_to_sdf: Path | str) -> BlockList:
    """Load sdf file into sdf_helper object.

    - if str is passed, it is converted to Path

    Args:
        path_to_sdf (Path | str): path to sdf file

    Raises:
        FileNotFoundError: file not found
        ValueError: file is not an sdf

    Returns:
        sdf_helper_internal: sdf data object
    """
    if not isinstance(path_to_sdf, Path):
        path_to_sdf = Path(path_to_sdf)
    if not path_to_sdf.exists():
        raise FileNotFoundError(f"File not found: {path_to_sdf}")
    if not path_to_sdf.suffix == ".sdf":
        raise ValueError(f"File is not an sdf: {path_to_sdf}")

    data = sh.getdata(str(path_to_sdf), verbose=False)
    return data


@dataclass
class SDF:
    grid: Grid
    mid_grid: Grid
    mag: Mag
    numberdensity: np.ndarray
    temperature: np.ndarray


def load_sdf_verified(path_to_sdf: Path) -> SDF:
    data = load(path_to_sdf=path_to_sdf)
    return SDF(
        grid=get_grid(data, mid=False),
        mid_grid=get_grid(data, mid=True),
        mag=get_mag(data),
        numberdensity=data.Derived_Number_Density.data,
        temperature=data.Derived_Temperature.data,
    )


def extract_vars(data, vars_list: list[str]) -> dict[str, np.ndarray]:
    """Get vars from `vars_list` out of sdf data object.

    Args:
        data (sdf_helper_internal): sdf data object
        vars_list (list[str]): list of variable names

    Returns:
        dict[str, np.ndarray]: dict of np arrays, keys are var names
    """
    out = {}
    grid_name = "Grid_Grid_mid"
    for var in vars_list:
        dat = getattr(data, var).data
        if var == grid_name:
            dat = dat[0]  # only care about x
        out[var] = dat

    return out


def reduce_to_1d(data: dict[str, np.ndarray], index: int = 80) -> dict[str, np.ndarray]:
    """Take y=`index` from each key in dict of 2d np arrays.

    Args:
        data (dict[str, np.ndarray]): dict of 2d np arrays
        index (int, optional): y=`index`. Defaults to 80.

    Raises:
        ValueError: Raised when any key in dict has dimension != 2

    Returns:
        dict[str, np.ndarray]: dict of 1d np arrays
    """
    out = {}
    for k, v in data.items():
        print(k, type(v), v.shape)
        if len(v.shape) == 2:
            out[k] = v[:, index]
        elif len(v.shape) == 1:
            out[k] = v
        else:
            raise ValueError(
                f"{k} has too many/not enough dimensions. Has shape {v.shape}"
            )

    return out


def data_to_df(data: dict[str, np.ndarray]) -> pd.DataFrame:
    """Turn dict into dataframe.

    Args:
        data (dict[str, np.ndarray]): dict

    Returns:
        pd.DataFrame: dataframe
    """
    return pd.DataFrame(data)


def save_csv(path: Path, data: pd.DataFrame) -> None:
    """Save dataframe as csv at `path`.

    Args:
        path (Path): Path (inc. filename) to output
        data (pd.DataFrame): dataframe to write
    """
    data.to_csv(path, index=False)


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
    if mid:
        attr = data.Grid_Grid_mid.data
    else:
        attr = data.Grid_Grid.data
    return Grid(x=attr[0], y=attr[1])


def get_mag(data) -> Mag:
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
    return Mag(
        bx=data.Magnetic_Field_Bx.data,
        by=data.Magnetic_Field_By.data,
        bz=data.Magnetic_Field_Bz.data,
    )
