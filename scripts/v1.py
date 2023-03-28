"""Plot variables from `.sdf` file."""
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import sdf_helper as sh
from epoch_cheats import get_deck_constants
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
    """Plot numberdensity into axis `ax`.

    Also sets x and y axis labels.

    Args:
        ax (plt.Axes): axes to plot onto
        x (np.ndarray): x axis data
        ndens (np.ndarray): y axis data
    """
    x, ndens = hj.trim_vars([x, ndens], slice(None, -5))
    ax.plot(x, ndens)
    ax.set_ylabel(hj.make_label(hj.Label.ND, si=True))
    ax.set_xlabel(hj.make_label(hj.Label.X, si=True))


def collapse_bxyz(bxyz: hj.Mag) -> hj.Mag:
    """Take mean of each component of bxyz in spatial y axis.

    Args:
        bxyz (hj.Mag): magnetic field (bx:[x,y], ...)

    Returns:
        hj.Mag: Magnetic field (bx:[x], ...)
    """
    out = []
    for b in bxyz:
        out.append(b.mean(axis=1))
    return hj.Mag._make(out)


def plot_bxyz(ax: plt.Axes, x: np.ndarray, bxyz: hj.Mag) -> None:
    """Plot (1d) magnetic field components bt,bx,by,bz.

    Args:
        ax (plt.Axes): axis to plot onto
        x (np.ndarray): x axis values
        bxyz (hj.Mag): components of magnetic field
    """
    slc = slice(None, -5)
    x = x[slc]
    ax.plot(x, np.linalg.norm([i for i in bxyz], axis=0)[slc])
    names = [f"$b_{i}$" for i in "xyz"]
    for c, n in zip(bxyz, names):
        c = c[slc]
        ax.plot(x, c, label=n)


def grad_ndens(
    x: np.ndarray, ndens: np.ndarray, trim: int = -5
) -> tuple[np.ndarray, np.ndarray]:
    """Compute gradient of numberdiensity.

    Args:
        x (np.ndarray): x axis values
        ndens (np.ndarray): numberdensity
        trim (int, optional): number of values to trim off end of x. Defaults to -5.

    Returns:
        tuple[np.ndarray, np.ndarray]: (trimmed x, trimmed d/dx(nd))
    """
    xt, g_ndenst = hj.trim_vars(
        [gridmid.x, np.gradient(ndens, gridmid.x[1] - gridmid.x[0])],
        slice(None, trim),
    )
    return xt, g_ndenst


def rolling_avg(x: np.ndarray, y: np.ndarray, width: float) -> np.ndarray:
    """Calculate a rolling average of `width` (in `x` units) over `y`.

    Args:
        x (np.ndarray): x values
        y (np.ndarray): y values
        width (float): width of window to avg over (units of `x`)

    Returns:
        np.ndarray: averaged `y`, length of original `y` is retained
    """
    dx: float = x[1] - x[0]
    w = int(width // dx)
    out = np.convolve(y, np.ones(w), "same") / w
    return out


if __name__ == "__main__":
    data_dir = Path("U6T40")
    constants = get_deck_constants(data_dir / "input.deck")
    print(f"Ion inertial length: {constants['di']} m/s")
    data = sh.getdata(str(data_dir / "0128.sdf"), verbose=False)
    ndens = data_collapse_y(data, "Derived_Number_Density")
    bxyz = hj.get_mag(data)
    gridmid = hj.get_grid(data, mid=True)

    fig, axs = plt.subplots(2, 1, sharex=True)
    plot_ndens(axs[0], x=gridmid.x, ndens=ndens)
    xt, g_ndenst = grad_ndens(gridmid.x, ndens)
    g_ndenstr = rolling_avg(xt, g_ndenst, constants["di"])
    axs[1].plot(xt, g_ndenstr)

    plt.legend()
    plt.tight_layout()
    plt.show()
