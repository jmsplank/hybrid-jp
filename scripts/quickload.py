from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sdf_helper as sh
from rich import print

import hybrid_jp as hj


def load(path_to_sdf: Path):
    data = sh.getdata(str(path_to_sdf), verbose=False)
    return data


def extract_vars(data, vars_list: list[str]) -> dict[str, np.ndarray]:
    out = {}
    grid_name = "Grid_Grid_mid"
    for var in vars_list:
        dat = getattr(data, var).data
        if var == grid_name:
            dat = dat[0]  # only care about x
        out[var] = dat

    return out


def reduce_to_1d(data: dict[str, np.ndarray], index: int = 80) -> dict[str, np.ndarray]:
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
    return pd.DataFrame(data)


def save_csv(path: Path, data: pd.DataFrame) -> None:
    data.to_csv(path, index=False)


def main(i: int):
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
