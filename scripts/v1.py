from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sdf_helper as sh
from phdhelper.helpers import override_mpl

import hybrid_jp as hj

override_mpl.override("|krgb")


def data_collapse_y(data, var: str) -> np.ndarray:
    """Get mean of variable in 2d data along y axis.

    Args:
        data (sdf.BlockList): data from sh.get_data()
        var (str): Variable name from sh.list_variables(data)

    Returns:
        np.ndarray: 1d averaged data along x axis
    """
    item = getattr(data, var).data
    item = cast(np.ndarray, item)

    item = np.mean(item, axis=1)

    return item


def plot_ndens(ax: plt.Axes, x: np.ndarray, ndens: np.ndarray) -> None:
    x, ndens = hj.trim_vars([x, ndens], slice(None, -5))
    ax.plot(x, ndens)
    ax.set_ylabel(hj.make_label(hj.Label.ND, si=True))
    ax.set_xlabel(hj.make_label(hj.Label.X, si=True))


def collapse_bxyz(bxyz: hj.Mag) -> hj.Mag:
    out = []
    for b in bxyz:
        out.append(b.mean(axis=1))
    return hj.Mag._make(out)


def plot_bxyz(ax: plt.Axes, x: np.ndarray, bxyz: hj.Mag) -> None:
    slc = slice(None, -5)
    x = x[slc]
    ax.plot(x, np.linalg.norm([i for i in bxyz], axis=0)[slc])
    for c in bxyz:
        c = c[slc]
        ax.plot(x, c)


if __name__ == "__main__":
    data_dir = Path("U6T40")
    data = sh.getdata(str(data_dir / "0199.sdf"), verbose=False)
    ndens = data_collapse_y(data, "Derived_Number_Density")
    bxyz = hj.get_mag(data)
    gridmid = hj.get_grid(data, mid=True)
    fig, axs = plt.subplots(2, 1)
    plot_ndens(axs[0], x=gridmid.x, ndens=ndens)
    plot_bxyz(axs[1], x=gridmid.x, bxyz=collapse_bxyz(bxyz))

    plt.tight_layout()
    plt.show()
