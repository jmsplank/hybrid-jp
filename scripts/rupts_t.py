"""Simple srcipt to test functionality of `ruptures` package on simulation data.

Loads data as `.csv` from scripts/data/xxxx.csv. To generate this from `.sdf` files
run `scripts/quickload.py` first.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from phdhelper.helpers import override_mpl

import hybrid_jp as hj


def load_csv(path: Path) -> pd.DataFrame:
    """Load a csv file from `path`.

    Args:
        path (Path): path to `.csv` representation of `.sdf` data file

    Returns:
        pd.DataFrame: pandas df of csv data
    """
    return pd.read_csv(path, header=0)


def df_to_rows_array(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert dataframe into 2d np array [x, columns].

    Args:
        data (pd.DataFrame): data
        cols (list[str]): Column names to transform into array

    Returns:
        np.ndarray: 2d array of data
    """
    arr = np.empty((len(data), len(cols)))
    for i, col in enumerate(cols):
        arr[:, i] = data[col].values
    return arr


def rupture_algo_pelt(arr: np.ndarray, penalty: int = 20) -> list[float]:
    """Use Penalised Detection Point algorithm to detect change points.

    https://centre-borelli.github.io/ruptures-docs/code-reference/detection/pelt-reference/#ruptures.detection.pelt.Pelt.fit

    Args:
        arr (np.ndarray): 2d array of [x, features]
        penalty (int, optional): penalty value to determine number of change points. Defaults to 20.

    Returns:
        list[float]: index of chage points along first axis of array
    """
    print("Algorithm-ing")
    algo = rpt.Pelt(model="rbf").fit(arr)
    result = algo.predict(pen=penalty)
    print("Found some state transitions:")
    print(result)
    return result


def rupture_algo_binseg(arr: np.ndarray, nseg=3) -> list[int]:
    """Use Binary Segmentation algorithm to detect change points.

    Args:
        arr (np.ndarray): 2d array of [x,features]
        nseg (int, optional): number of change points to return. Defaults to 3.

    Returns:
        list[int]: index of change points
    """
    print("Binary Segmentation")
    algo = rpt.Binseg(model="l2", min_size=2).fit(arr)
    result = algo.predict(n_bkps=nseg)
    print("Found some state transitions:")
    print(result)
    return result


def plot(
    data: pd.DataFrame,
    x_name: str,
    lines: list[int] | None = None,
    save_path: Path | None = None,
) -> None:
    """Plot data.

    Args:
        data (pd.DataFrame): The data to plot
        x_name (str): column name representing x-axis values
        lines (list[int] | None, optional): indices of change points. Defaults to None.
        save_path (Path | None, optional): path to save the resulting `.png`. Defaults to None.
    """
    cols = list(data)
    cols.remove(x_name)
    n_plots = len(cols)
    fig, axs = plt.subplots(n_plots, 1, sharex=True)

    for i in range(n_plots):
        axs[i].plot(data[x_name], data[cols[i]])

    if lines is not None:
        if len(data) in lines:
            lines.remove(len(data))
        for li in lines:
            for i in range(n_plots):
                axs[i].axvline(data[x_name].values[li], color="r", ls="--")  # type: ignore

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    if save_path:
        axs[0].set_title(save_path.stem)
        plt.savefig(save_path, dpi=300)


def main(
    fnumber: int,
    save_name: str | None = None,
    save_path: Path = Path("scripts/temp/plots"),
):
    """Entry point.

    Args:
        fnumber (int): frame number of `.sdf` file
        save_name (str | None, optional): name of `.png` file. Defaults to None.
        save_path (Path, optional): location to save `.png` plot. Defaults to Path("scripts/temp/plots").
    """
    csv = Path(f"scripts/data/{fnumber:04d}.csv")
    data = load_csv(csv)
    print(data.head())
    arr = df_to_rows_array(
        data,
        [
            "Derived_Number_Density",
            "Magnetic_Field_Bx",
            "Magnetic_Field_By",
            "Magnetic_Field_Bz",
        ],
    )
    transitions = rupture_algo_binseg(arr, nseg=2)
    if save_name is None:
        save_name = f"i{fnumber:04d}_changes.png"
    plot(data, "Grid_Grid_mid", lines=transitions, save_path=save_path / save_name)


if __name__ == "__main__":
    override_mpl.override("|krgb")
    for i in [10, 50, 100, 130, 150]:
        print(i)
        main(i)
