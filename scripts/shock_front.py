"""Finds shock front and change points.

Shock front is found by finding the peak of the gradient of the number density.
change points are found by binseg.
"""
from collections.abc import Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epoch_cheats import evaluate_deck
from matplotlib.colors import LogNorm
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import blue, green, red
from rich import print as rprint
from sdf import BlockList
from tqdm import tqdm

import hybrid_jp as hj


def find_shock_index_from_gradnd(gradnd: np.ndarray) -> int:
    """Find the index of the shock from gradient of the number density.

    - If gradnd is 2d then find the peak of the mean.
    - Assume the shock is located at the peak of grad nd

    Args:
        gradnd (np.ndarray): gradient of the number density.

    Returns:
        int: index of the shock.
    """
    if gradnd.ndim == 2:
        gradnd = gradnd.mean(axis=1)

    return int(np.argmax(gradnd))


def plot_nd(x: np.ndarray, nd: np.ndarray):
    """Plot the number density and its gradient."""
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(x, nd)
    axs[1].plot(x, np.gradient(nd))

    # axis labels
    axs[0].set_ylabel("$n$ (m$^{-3}$)")
    axs[1].set_ylabel(r"$\frac{d}{dx} (n)$")
    axs[1].set_xlabel("$x$ (m)")

    plt.tight_layout()
    # plt.show()


def iter_over_all_sdfs(skip_at_start: int = 0) -> Iterator[Path]:
    """Iterate over all the sdfs in the data directory."""
    data_dir = Path("U6T40")
    sdfs = data_dir.glob("*.sdf")
    sorted_sdfs = sorted(sdfs, key=lambda p: int(p.stem))
    sorted_sdfs = sorted_sdfs[skip_at_start:]
    for sdf_file in sorted_sdfs:
        yield sdf_file


def get_shock_and_change_point_indices(
    path_to_sdf: Path,
) -> tuple[np.ndarray, int, list[int]]:
    """Get the shock index and change point indices.

    Args:
        path_to_sdf (Path): path to sdf file

    Returns:
        tuple[np.ndarray, int, list[int]]: (bt, shock_index, change_points)
    """
    data = hj.sdf_files.load(path_to_sdf)
    trim = slice(None, -10)
    nd_all_y = data.Derived_Number_Density.data[trim, :]
    nd = cast(np.ndarray, nd_all_y.mean(axis=1))

    shock_index = find_shock_index_from_gradnd(np.gradient(nd))
    change_points = hj.binseg(nd_all_y)

    mag = np.stack(list(hj.sdf_files.get_mag(data)), 0)
    bt = np.linalg.norm(mag, axis=0).mean(axis=1)

    return bt, shock_index, change_points


def get_number_of_files() -> int:
    """Get the number of sdf files in the U6T40 directory.

    Returns:
        int: number of files
    """
    return len(list(Path("U6T40").glob("*.sdf")))


def main():
    """Entry point."""
    override_mpl.override("|krgb")

    deck = evaluate_deck(Path("U6T40/input.deck"))
    dt = deck["output"]["dt_snapshot"]
    t_end = deck["control"]["t_end"]
    if not isinstance(dt, float) or not isinstance(t_end, float):
        raise TypeError("dt and end must be floats")

    t = np.arange(0, t_end + dt, dt)
    data = hj.sdf_files.load(Path("U6T40/0000.sdf"))
    grid = hj.get_grid(data, mid=False)
    grid_mid = hj.get_grid(data, mid=False)

    num_files = get_number_of_files()
    skip = 10
    num_steps = num_files - skip

    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap(
                    get_shock_and_change_point_indices,
                    iter_over_all_sdfs(skip_at_start=skip),
                ),
                total=num_steps,
                desc="Calculating shock and change point index",
            )
        )
        bts, shocks_i, c_points_i = map(list, zip(*results))

    btarray = np.array(bts)
    print(btarray.shape)
    # plot pcolormesh of the number density
    xx, tt = np.meshgrid(grid.x, t[skip - 1 :] + dt / 2)
    plt.pcolormesh(xx, tt, btarray)

    plt.plot(grid_mid.x[shocks_i], t[skip:])

    changes = np.array(c_points_i)
    for i in range(3):
        ls = ["--", "-.", ":"][i]
        plt.plot(grid_mid.x[changes[:, i]], t[skip:], color="w", linestyle=ls)

    xlim = [min(grid_mid.x), max(grid_mid.x)]
    plt.xlim(xlim)
    plt.ylim((0, t_end))

    plt.xlabel("$x$ (m)")
    plt.ylabel("$t$ (s)")
    plt.colorbar(label="$|B|$ (T)")

    plt.tight_layout()
    plt.show()

    print("writing to csv")
    data = {}
    data["t"] = t[skip:]
    data["shock"] = grid_mid.x[shocks_i]
    for i in range(3):
        data[f"change_{i}"] = grid_mid.x[changes[:, i]]
    df = pd.DataFrame(data)
    df.to_csv(Path("scripts") / "shock_changes.csv", index=False)
    print("done.")


if __name__ == "__main__":
    main()
