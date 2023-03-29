from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from epoch_cheats import evaluate_deck
from phdhelper.helpers import override_mpl
from rich import print as rprint

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


def iter_over_all_sdfs(skip_at_start: int = 0):
    """Iterate over all the sdfs in the data directory."""
    data_dir = Path("U6T40")
    sdfs = data_dir.glob("*.sdf")
    sorted_sdfs = sorted(sdfs, key=lambda p: int(p.stem))
    sorted_sdfs = sorted_sdfs[skip_at_start:]
    for sdf_file in sorted_sdfs:
        yield hj.sdf_files.load(sdf_file)


if __name__ == "__main__":
    override_mpl.override("|krgb")

    deck = evaluate_deck(Path("U6T40/input.deck"))
    dt = deck["output"]["dt_snapshot"]
    t_end = deck["control"]["t_end"]
    if not isinstance(dt, float) or not isinstance(t_end, float):
        raise TypeError("dt and end must be floats")

    t = np.arange(0, t_end + dt, dt)

    indices = []
    skip = 10
    for i, data in enumerate(iter_over_all_sdfs(skip_at_start=skip)):
        trim = slice(None, -10)
        if i == 0:
            x = hj.get_grid(data, mid=True).x[trim]
        nd = data.Derived_Number_Density.data[trim, :].mean(axis=1)

        shock_index = find_shock_index_from_gradnd(np.gradient(nd))
        print(f"Shock index: {shock_index}")
        indices.append(shock_index)

    plt.plot(x[indices], t[skip:])  # type: ignore
    xlim = [min(x), max(x)]  # type: ignore
    plt.xlim(xlim)
    plt.ylim((0, t_end))

    plt.tight_layout()
    plt.show()
