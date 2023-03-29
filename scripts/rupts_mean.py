"""Detect change points using binary segmentation on the gradient of the number density.

Uses all y values in grad_nd as features for the binary segmentation algorithm.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sdf
from matplotlib.colors import LogNorm
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import red
from v1 import collapse_bxyz

from hybrid_jp import Grid, Mag, binseg, df_to_rows_array
from hybrid_jp.sdf_files import get_grid, get_mag, load


def get_grad_nd(data: sdf.BlockList) -> np.ndarray:
    # google style docstring
    """Get the gradient of the number density along the x axis.

    Args:
        data (sdf.BlockList): sdf data object to get gradient of.

    Returns:
        np.ndarray: gradient of the number density.
    """
    nd = data.Derived_Number_Density.data
    grad_nd = np.gradient(nd, axis=0)
    return grad_nd


def plot_meshgrid_of_gradnd(gradnd: np.ndarray, grid_edges: Grid) -> None:
    """Plot the meshgrid of the gradient of the number density.

    Requires the grid edges to be passed in since pcolormesh requires the
    edges of the grid to be one larger than the number of grid points in
    each dimension.

    Args:
        gradnd (np.ndarray): gradient of the number density.
        grid (Grid): Grid namedtuple of x and y grid values.
    """
    X, Y = np.meshgrid(grid_edges.x, grid_edges.y)
    plt.pcolormesh(X, Y, gradnd.T, norm=LogNorm())
    plt.colorbar(label="$d/dx n_d$")
    plt.tight_layout()
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    save_dir = Path("scripts/temp/rupts_mean/gradnd_meshgrid.png")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dir, dpi=300)
    plt.clf()
    plt.close()


def plot_1d_with_change_points(
    x: np.ndarray,
    nd_1d: np.ndarray,
    grad_nd_1d: np.ndarray,
    mag_1d: Mag,
    change_points: list[int],
    trim: slice = slice(None, -5),
) -> None:
    # make a figure with 3 horizontal subplots
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 6))
    # plot the number density on axs[0]
    axs[0].plot(x[trim], nd_1d[trim])
    # plot the gradient of the number density on axs[1]
    axs[1].plot(x[trim], grad_nd_1d[trim])
    # plot the magnetic field by and bz on axs[2]
    axs[2].plot(x[trim], mag_1d.by[trim], label="$B_y$")
    axs[2].plot(x[trim], mag_1d.bz[trim], label="$B_z$")

    # plot the change points on all subplots
    for ax in axs:
        for cp in change_points:
            ax.axvline(x[cp], color=red, linestyle="--")

    # set the labels and titles
    axs[0].set_ylabel("$n_d$ (m$^{-3}$)")
    axs[1].set_ylabel(r"$\frac{d}{dx} (n_d)$")
    axs[2].set_ylabel("$B$ (T)")
    axs[2].set_xlabel("x (m)")
    axs[0].set_title("Change points in $n_d$, $d/dx n_d$, and $B_y$, $B_z$")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


def mean_of_var(var: sdf.BlockPlainVariable) -> np.ndarray:
    """Get the mean of a variable.

    Args:
        var (sdf.BlockPlainVariable): sdf variable to get mean of.

    Returns:
        np.ndarray: mean of the variable.
    """
    return var.data.mean(axis=1)


if __name__ == "__main__":
    override_mpl.override("|krgb")
    data = load("U6T40/0128.sdf")
    grid_edges = get_grid(data, mid=False)
    grid_mid_x = get_grid(data, mid=True).x
    grad_nd = get_grad_nd(data)
    plot_meshgrid_of_gradnd(grad_nd, grid_edges)

    trim = slice(None, -10)
    change_points = binseg(data.Derived_Number_Density.data[trim, :], nseg=3)
    print(f"Change points: {change_points}")
    grad_nd_1d = grad_nd.mean(axis=1)
    nd_1d = mean_of_var(data.Derived_Number_Density)
    mag = get_mag(data)
    mag_1d = collapse_bxyz(mag)
    plot_1d_with_change_points(
        grid_mid_x,
        nd_1d,
        grad_nd_1d,
        mag_1d,
        change_points,
        trim,
    )
