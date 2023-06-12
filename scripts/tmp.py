from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sdf_helper as sh
from phdhelper.helpers import override_mpl

import hybrid_jp as hj


def plot(grid: hj.Grid, data: np.ndarray):
    fig, ax = plt.subplots()

    XX, YY = np.meshgrid(grid.x, grid.y)

    ax.pcolormesh(XX, YY, data.T)
    plt.tight_layout()
    plt.show()


def main(path: Path):
    data = sh.getdata(str(path), verbose=False)
    grid = hj.get_grid(data, mid=False)
    ndens = data.Derived_Number_Density.data
    plot(grid, ndens)


if __name__ == "__main__":
    override_mpl.override()
    frame = Path("U6T40/0128.sdf")
    main(path=frame)
