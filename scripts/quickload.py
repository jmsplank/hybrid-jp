"""Functions to extract `.sdf` file into `.csv`."""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sdf_helper as sh
from rich import print

import hybrid_jp as hj
from hybrid_jp.sdf_files import data_to_df, extract_vars, load, reduce_to_1d, save_csv


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
