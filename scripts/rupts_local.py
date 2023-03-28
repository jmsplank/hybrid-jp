"""Use the ruptures package to find change points in simulated shock crossing.

Loads data as `.csv` from scripts/data/xxxx.csv. To generate this from `.sdf` files
run `scripts/quickload.py` first.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
from epoch_cheats import get_deck_constants
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import green
from quickload import extract_vars, load
from rich import print

import hybrid_jp as hj

override_mpl.override()


def extract_row(
    index: int,
    data: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Get the same row of 2d array from every key in dict.

    Args:
        index (int): the row to get
        data (dict[str, np.ndarray]): dict of 2d np arrays

    Returns:
        dict[str, np.ndarray]: dict of 1d np arrays corresponding to index `[:,i]` of input array
    """
    cols = list(data)
    out = {}
    for c in cols:
        out[c] = data[c][:, index]
    return out


def mean_allrows(
    data: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Get mean & std of dict of 2d np arrays.

    Args:
        data (dict[str, np.ndarray]): dict of 2d np arrays

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]: (dict of means, dict of std's)
    """
    cols = list(data)
    out = {}
    out_sd = {}
    for c in cols:
        out[c] = np.mean(data[c], axis=1)
        out_sd[c] = np.std(data[c], axis=1)
    return out, out_sd


def drop_x_from_dict(old_dict: dict, x_name: str = "Grid_Grid_mid") -> dict:
    """Remove key from dict that matches `x_name.

    Args:
        old_dict (dict): input dictionary
        x_name (str, optional): key value representing x data. Defaults to "Grid_Grid_mid".

    Returns:
        dict: dict missing x values
    """
    new_dict = dict(old_dict)
    del new_dict[x_name]
    return new_dict


def dict_rows_to_array(data: dict[str, np.ndarray], xlen: int) -> np.ndarray:
    """Collapse dict of 1d arrays to 2d np array.

    Args:
        data (dict[str, np.ndarray]): dict of 1d arrays
        xlen (int): length of each array (must all be same length)

    Returns:
        np.ndarray: 2d array where y axis is each key in input dict
    """
    cols = list(data)
    n_cols = len(cols)
    out = np.empty((xlen, n_cols))
    for i in range(n_cols):
        out[:, i] = data[cols[i]]
    return out


def plot(
    x: np.ndarray,
    y: dict[str, np.ndarray],
    y_sd: dict[str, np.ndarray],
    lines: list[list[float]],
    save_name: str,
) -> None:
    """Plot the rsesult.

    Args:
        x (np.ndarray): the x axis (shared)
        y (dict[str, np.ndarray]): y axis variables
        y_sd (dict[str, np.ndarray]): sd of each variable
        lines (list[list[float]]): x coordinate of detected change points
        save_name (str): name to save results as into scripts/temp/plots
    """
    fig, axs = plt.subplots(len(y) + 1, 1, sharex=True, figsize=(10, 8))

    vars = list(y)
    skip = 100
    flatlines = [line for subline in lines for line in subline]
    for i, var in enumerate(vars):
        for li in flatlines:
            axs[i].axvline(li, color=green, alpha=0.5)
        axs[i].errorbar(
            x[::skip],
            y[var][::skip],
            yerr=y_sd[var][::skip],
            capsize=3,
            linestyle="none",
        )
        axs[i].plot(x, y[var], color="k")
        axs[i].set_ylabel(var)

    bins = 24
    axs[-1].hist(flatlines, bins=bins, density=True, color=green)
    axs[-1].set_yscale("log")

    axs[-1].set_xlabel("$x/d_i$")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(Path("scripts/temp/plots") / (f"{bins:03d}" + save_name), dpi=300)


def main(time_step: int):
    """Entry point.

    Args:
        time_step (int): Time step to load data from.
    """
    data = load(Path(f"U6T40/{time_step:04d}.sdf"))
    consts = get_deck_constants(Path("U6T40/input.deck"))
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
    data["d/dx nd"] = np.gradient(data["Derived_Number_Density"], axis=0)
    ynum = data["Derived_Number_Density"].shape[1]  # 160
    xname = "Grid_Grid_mid"
    ydata = drop_x_from_dict(old_dict=data, x_name=xname)
    x = data[xname] / consts["di"]
    all_changes = []
    for i in range(ynum):
        data_1d = extract_row(i, ydata)
        arr = dict_rows_to_array(data_1d, len(x))
        # arr = data_1d["d/dx nd"]
        i_changes = hj.binseg(arr, nseg=3)
        x_changes = [x[c] for c in i_changes if c != len(x)]
        all_changes.append(x_changes)
    y_me, y_sd = mean_allrows(ydata)
    plot(x, y_me, y_sd, all_changes, f"i{time_step:04d}_change_points_gradn.png")


if __name__ == "__main__":
    for i in [10, 50, 100, 130, 150]:
        main(i)
