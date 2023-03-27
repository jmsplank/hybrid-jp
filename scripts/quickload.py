"""Functions to extract `.sdf` file into `.csv`."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sdf_helper as sh
from rich import print

import hybrid_jp as hj


def load(path_to_sdf: Path):
    """Load data.

    Args:
        path_to_sdf (Path): Path to `.sdf`

    Returns:
        sdf_helper_internal: data
    """
    data = sh.getdata(str(path_to_sdf), verbose=False)
    return data


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


def main(i: int):
    """Entry point.

    Args:
        i (int): `.sdf` frame number`
    """
    data = load(Path(f"U6T40/{i:04d}.sdf"))
    data = extract_vars(
        data,
        [
            "Derived_Number_Density",
            "Grid_Grid_mid",
            "Magnetic_Field_Bx",
            "Magnetic_Field_By",
            "Magnetic_Field_Bz",
        ],
    )
    data = reduce_to_1d(data)
    data = data_to_df(data)
    print(data.head())
    save_csv(Path(f"scripts/data/{i:04d}.csv"), data)


if __name__ == "__main__":
    for i in [10, 50, 100, 130, 150]:
        print(i)
        main(i)
