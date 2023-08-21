"""Change point experiments.

The goal is to discover the best method/dataset or set of transformations
that will most accurately predict transitions from solar wind to shock
transition region to magnetosheath.
"""
# %% Importts
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ruptures as rpt
from cycler import cycler
from epoch_cheats import evaluate_deck, validate_deck
from matplotlib.collections import QuadMesh
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import MultipleLocator
from phdhelper import colours, mpl
from scipy.stats import norm as spnorm
from tqdm import tqdm

from hybrid_jp.sdf_files import load_sdf_verified

mpl.format()

# %% Load data
DATA_PATH = Path().resolve().parent / "U6T40"
fname = DATA_PATH / "0160.sdf"

sdf = load_sdf_verified(fname)
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %%

grid = sdf.mid_grid * (1 / deck.constant.di)
rows = dict(nd=sdf.numberdensity) | sdf.mag.all | sdf.elec.all | sdf.current.all
# del rows["bx"]

axs: list[plt.Axes]
fig, axs = plt.subplots(len(rows), 1, sharex=True, sharey=True, figsize=(7, 10))  # type: ignore

for i, (n, r) in enumerate(rows.items()):
    # axs[i].plot(grid.x, r.mean(axis=1), label=n)
    norm = {}
    if r.min() < 0:
        cmap = mpl.Cmaps.diverging
        rng = max(r.max(), -r.min())
        # norm["vmin"] = -rng
        # norm["vmax"] = rng
        norm["norm"] = SymLogNorm(vmin=-rng, vmax=rng, linthresh=0.005)  # type: ignore
    else:
        cmap = mpl.Cmaps.sequential

    axs[i].pcolormesh(grid.x, grid.y, r.T, rasterized=True, cmap=cmap, **norm)
    axs[i].set_ylabel(n)


plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()


# %%
def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise the array between 0 and 1."""
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))


fig, ax = plt.subplots()
cols = cycler("color", colours.sim.colours_hex + ["red", "green", "blue"])()
for i, (n, r) in enumerate(rows.items()):
    x = normalise(r[:, :10])
    col = next(cols)
    for i in range(x.shape[1]):
        _ = ax.plot(grid.x, x[:, i] - x[0, i], **col, alpha=0.2, ls="-")
        if i == 0:
            _[0].set_label(n)

ax.legend()
plt.tight_layout()
plt.show()

# %%


def plot_bp_alternating(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bps: list[int],
    label: str,
) -> QuadMesh:
    """Plots a 2D signal with alternating colors for each segment between breakpoints.

    Args:
        ax (plt.Axes): The axes on which to plot the signal.
        x (np.ndarray): The x-coordinates of the signal 1d.
        y (np.ndarray): The y-coordinates of the signal 1d.
        z (np.ndarray): The z-coordinates of the signal 2d.
        bps (list[int]): The indices of the breakpoints in the signal.
    """

    rng = max(z.max(), -z.min())
    style = {}
    if z.min() < 0:
        style["cmap"] = mpl.Cmaps.diverging
        style["vmin"] = -rng
        style["vmax"] = rng
    else:
        style["cmap"] = mpl.Cmaps.sequential
        style["vmin"] = 0
        style["vmax"] = rng
    ax.grid(False)
    im = ax.pcolormesh(x, y, z.T, **style)
    ax.axvline(x[bps[0]], color="k")
    ax.axvline(x[bps[1]], color="k")

    centres = [0] + bps + [len(x) - 1]
    centres = [(x[centres[i]] + x[centres[i + 1]]) / 2 for i in range(len(centres) - 1)]
    labels = ["SW", "STR", "MS"]
    y_height = y.max() - ((y.max() - y.min()) / 5)
    for i in range(3):
        ax.text(
            centres[i],
            y_height,
            labels[i],
            ha="center",
            va="center",
            fontsize=8,
            color="k",
        )
    ax.set_ylabel(label)

    return im


# %% Find three changepoints in jz

dx = grid.x[1] - grid.x[0]  # in d_i

model = "l2"
min_size = 10 // dx  # 10di in cells
n_breakpoints = 2
algo = rpt.Dynp(
    model=model,
    min_size=min_size,
).fit(sdf.current.jz)
breakpoints: list[int] = algo.predict(n_bkps=n_breakpoints)[:-1]  # type: ignore

fig, ax = plt.subplots()


im = plot_bp_alternating(ax, grid.x, grid.y, sdf.current.jz, breakpoints, "jz")
plt.colorbar(im, ax=ax, pad=0)
plt.show()

# %% Do it for all of them


def get_bps_dynp(
    z: npt.NDArray[np.float64], min_size: int = 66, model: str = "l2"
) -> list[int]:
    """Get breakpoints in z."""
    algo = rpt.Dynp(
        model=model,
        min_size=min_size,
    ).fit(z)
    return algo.predict(n_bkps=2)[:-1]  # type: ignore


def get_bps_binseg(
    z: npt.NDArray[np.float64],
    min_size: int = 66,
    model: str = "ar",
) -> list[int]:
    """Get breakpoints in z using binary segmentation.

    Args:
        z (npt.NDArray[np.float64]): data.
        min_size (int, optional): smallest interval between change points. Defaults to 66.
        model (str, optional): The model to use. Defaults to "ar".

    Returns:
        list[int]:  The indices of the breakpoints.
                    Excludes the last value from .predict() as that is the length of the array.
    """
    algo = rpt.Binseg(
        model=model,
        min_size=min_size,
    ).fit(z)
    return algo.predict(n_bkps=2)[:-1]  # type: ignore


def get_bps_win(
    z: npt.NDArray[np.float64],
    width: int = 66,
    model: str = "normal",
    jump: int | None = None,
) -> list[int]:
    """Use the window sliding method."""
    algo = rpt.Window(
        width=width,
        model=model,
        **(dict(jump=jump) if jump is not None else {}),
    ).fit(z)
    return algo.predict(n_bkps=2)[:-1]  # type: ignore


bp_dict = {}
for i, (n, r) in tqdm(enumerate(rows.items())):
    bps = get_bps_win(r, jump=10, model="ar")
    bp_dict[n] = bps

# %% Plot them all
fig, axs = plt.subplots(len(rows), 1, sharex=True, sharey=True, figsize=(7, 10))  # type: ignore

for i, (n, r) in enumerate(rows.items()):
    im = plot_bp_alternating(axs[i], grid.x, grid.y, r, bp_dict[n], n)
axs[0].set_title("Window Sliding using AR model")
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()

# %% Try inputting all the data at once to binseg

z_all = np.column_stack([normalise(r) for r in rows.values()]).astype(np.float64)

plt.pcolormesh(
    grid.x,
    np.arange(z_all.shape[-1]),
    z_all.T,
    rasterized=True,
    cmap=mpl.Cmaps.sequential,
)
plt.gca().yaxis.set_major_locator(MultipleLocator(160))

plt.show()

# %% All parameters at once

mod = "l1"
wid = 100
jmp = 5
bps = get_bps_win(z_all, width=wid, jump=jmp, model=mod)

lims = [0] + bps + [grid.x.size - 1]
regions: list[np.ndarray] = []
for i in range(3):
    data = z_all[lims[i] : lims[i + 1], :]
    regions.append(data)

# %%
widths = [grid.x[lims[i] : lims[i + 1]] for i in range(3)]
widths_size = [len(w) for w in widths]
widths_rel = [w / sum(widths_size) for w in widths_size]
fig, axs = plt.subplots(1, 3, sharey=True, gridspec_kw=dict(width_ratios=widths_rel))  # type: ignore

axs[1].set_xlabel(
    f"$x$ [$d_i$]\nWindow Sliding using {mod} model, width={wid}, jump={jmp}"
)

for i in range(3):
    axs[i].pcolormesh(
        widths[i],
        np.arange(z_all.shape[-1]),
        regions[i].T,
        rasterized=True,
        cmap=mpl.Cmaps.sequential,
    )
    axs[i].grid(False)
    axs[i].set_title(["SW", "STR", "MS"][i])
    axs[i].xaxis.set_major_locator(MultipleLocator(20))
    axs[i].yaxis.set_major_locator(MultipleLocator(160))

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.show()

# %%
fig, axs = plt.subplots(1, 3, sharey=True)  # type: ignore

for i, r in enumerate(regions):
    axs[i].grid(False)
    axs[i].hist(
        r.ravel(),
        bins=100,
        density=True,
        # edgecolor=colours.sim.dark(),
        # facecolor="none",
        alpha=0.5,
    )
    axs[i].set_title(["SW", "STR", "MS"][i])

    # Overplot a gaussian curve
    mu, sigma = np.mean(r), np.std(r)
    x = np.linspace(0, 1, 1000)
    y = spnorm.pdf(x, loc=mu, scale=sigma)
    axs[i].plot(x, y, color=colours.sim.red(), ls="--")

    # Set x axis ticks to mean += 1 sigma
    locs: list[float] = [mu - sigma, mu, mu + sigma]  # type: ignore
    axs[i].set_xticks(locs, labels=[f"{lab:.2f}" for lab in locs])
    for lab in locs:
        axs[i].axvline(lab, color="k", ls="-" if lab == mu else "--")

    if i == 1:
        axs[i].set_xlabel("Normalised")
    if i == 0:
        axs[i].set_ylabel("Density")

plt.tight_layout()
plt.subplots_adjust(wspace=0)
plt.show()

# %%
