"""Functions that interact with sdf_helper objects."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import sdf_helper as sh
from sdf import BlockList

from .dtypes import Current, Elec, Grid, Mag


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
    """Representation of .sdf file."""

    grid: Grid
    mid_grid: Grid
    mag: Mag
    elec: Elec
    current: Current
    numberdensity: np.ndarray
    temperature: np.ndarray
    tstamp: float | None = None  # Optional bc comes from .deck not .sdf


def load_sdf_verified(path_to_sdf: Path, dt: float | None = None) -> SDF:
    """Load an sdf into SDF class.

    Note:
        SDF.tstamp will be None if dt is None.
        Otherwise it will be the timestamp of the sdf file, created by multiplying
        dt with the filename.
    """
    data = load(path_to_sdf=path_to_sdf)
    return SDF(
        grid=get_grid(data, mid=False),
        mid_grid=get_grid(data, mid=True),
        mag=get_mag(data),
        elec=get_elec(data),
        current=get_current(data),
        numberdensity=data.Derived_Number_Density.data,
        temperature=data.Derived_Temperature.data,
        tstamp=int(path_to_sdf.stem) * dt if dt else None,
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
    data: BlockList,
    show_name: bool = True,
    show_type: bool = False,
    show_size: bool = True,
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
                params.append(str(type(val)))
            if show_size:
                params.append(np.array2string(np.array(val.dims), separator=","))
            out.append(params)

        except Exception:
            pass
    return out


def print_variables(
    data: BlockList,
    show_name: bool = True,
    show_type: bool = False,
    show_size: bool = True,
) -> None:
    """Print variables in an sdf_helper object.

    Args:
        data (BlockList): SDF file from sdf_helper.
        show_name (bool, optional): Show variable names. Defaults to True.
        show_type (bool, optional): Show variable types. Defaults to False.
        show_size (bool, optional): Show variable sizes. Defaults to True.

    Example:
        >>> import hybrid_jp
        >>> sdf = hybrid_jp.sdf_files.load("U6T40/0003.sdf")
        >>> hybrid_jp.sdf_files.print_variables(sdf)
        CPUs_Current_rank                   [0,0]
        CPUs_Original_rank                  [1,8]
        Current_Jx                          [1600, 160]
        Current_Jy                          [1600, 160]
        Current_Jz                          [1600, 160]
        Derived_Average_Particle_Energy     [1600, 160]
        ...
    """
    vars = list_variables(data, show_name, show_type, show_size)
    n_cols = len(vars[0])
    max_lens = []
    for i in range(n_cols):
        max_lens.append(max([len(v[i]) for v in vars]))

    for v in vars:
        strs = [v[i] + " " * (max_lens[i] - len(v[i])) for i in range(n_cols)]
        print("\t".join(strs))


def get_grid(data: BlockList, mid: bool = False) -> Grid:
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


def get_mag(data: BlockList) -> Mag:
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


def get_elec(data: BlockList) -> Elec:
    """Get the electric field of an sdf_helper object.

    Args:
        data (sdf_helper): sdf data object to get electric field of.

    Returns:
        Elec: Elec namedtuple of x, y, and z electric field values.

    Example:
        >>> get_elec(data)
        Elec(x=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,]],
            y=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]],
            z=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]]))
    """
    return Elec(
        ex=data.Electric_Field_Ex.data,
        ey=data.Electric_Field_Ey.data,
        ez=data.Electric_Field_Ez.data,
    )


def get_current(data: BlockList) -> Current:
    """Get the electric field of an sdf_helper object.

    Args:
        data (sdf_helper): sdf data object to get electric field of.

    Returns:
        Elec: Elec namedtuple of x, y, and z electric field values.

    Example:
        >>> get_elec(data)
        Elec(x=array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, ...,]],
            y=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]],
            z=array([[ 0.00000000e+00,  1.00000000e-16,  2.00000000e-16, ...,]]))
    """
    return Current(
        jx=data.Current_Jx.data,
        jy=data.Current_Jy.data,
        jz=data.Current_Jz.data,
    )


def filefinder(dir: Path, start: int, stop: int) -> Iterator[Path]:
    """Returns next sdf file in dir.

    Note:
        SDFs are named as 0000.sdf, 0001.sdf, etc. i.e. 0 padded 4 digit integers.

    Args:
        dir (Path): directory to search
        start (int): start file number
        stop (int): stop file number

    Yields:
        Iterator[Path]: next sdf file in dir

    Example:
        >>> dir = Path("data")
        >>> for file in filefinder(dir, 0, 10):
        ...     print(file)
        dir/0000.sdf
        dir/0001.sdf
        ...
    """
    yield from (dir / f"{i:04d}.sdf" for i in range(start, stop + 1))
