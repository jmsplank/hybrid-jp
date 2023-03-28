"""Detect change points using binary segmentation on the gradient of the number density.

Uses all y values in grad_nd as features for the binary segmentation algorithm.
"""
import matplotlib.pyplot as plt
import numpy as np
import sdf
from matplotlib.colors import LogNorm
from phdhelper.helpers import override_mpl

from hybrid_jp import Grid, binseg, df_to_rows_array
from hybrid_jp.sdf_files import get_grid, load


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
    plt.colorbar()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    override_mpl.override()
    data = load("U6T40/0128.sdf")
    grid_edges = get_grid(data, mid=False)
    grad_nd = get_grad_nd(data)
    plot_meshgrid_of_gradnd(grad_nd, grid_edges)
