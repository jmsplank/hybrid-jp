"""Follows from ./consistency_of_changes.py

The aim now has changed to accurately finding the shock/start of the transition region.
From this, the downstream will be broken up into small chunks and the end of the
transition region will be when the turbulence is fully developed.
"""
# %% Imports
import numpy.typing as npt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ruptures as rpt
from epoch_cheats import evaluate_deck, validate_deck
from scipy.stats import norm
from phdhelper import mpl
from phdhelper.colours import sim as colours
from multiprocessing import Pool, set_start_method
from functools import partial
from hybrid_jp.sdf_files import SDF, load_sdf_verified
from pathlib import Path
from typing import Literal, NamedTuple, Iterator
from scipy.optimize import curve_fit


# %% CONSTS
mpl.format()
set_start_method("fork")

DATA_PATH = Path().resolve().parent / "U6T40"
FSTART, FSTOP = 20, 200
N_THREADS = 7

n_files = FSTOP - FSTART + 1

# %% Load deck
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))


# %% Define a central data type that future functions are based on
def _recurse_sdf(obj, attrs: list[str]):
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


class Step(NamedTuple):
    """Container for one timestep of simulation.

    Holds data for a single .sdf file and some associated metadata.
    """

    tstamp: float
    sdf: SDF
    changes: dict[str, npt.NDArray[np.float64]] = {}

    def get_qty(self, qty: str) -> npt.NDArray[np.float64]:
        """Get a quantity from the sdf file."""
        attrs = qty.split(".")
        qty_arr = _recurse_sdf(self.sdf, attrs)
        if not isinstance(qty_arr, np.ndarray):
            raise TypeError(f"Quantity {qty} is not a numpy array.")
        return qty_arr


# %% Define funcitons for detecting changes and loading data
def detect_changes(
    method: Literal["binseg", "dynp", "win"],
    cost: Literal["l1", "l2", "ar", "norm"],
    arr: npt.NDArray[np.float64],
    n_bkps: int = 1,
    jump: int = 10,
    min_size: int = 70,
    win_width: int = 100,
) -> list[int]:
    """
    Detect changes in a signal using a change point detection algorithm.

    Args:
        method (Literal["binseg", "dynp", "win"]): The change point detection algorithm to use.
        cost (Literal["l1", "l2", "ar", "norm"]): The cost function to use for the algorithm.
        arr (npt.NDArray[np.float64]): The signal to detect changes in.
        n_bkps (int, optional): The number of change points to detect. Defaults to 1.
        jump (int, optional): The maximum distance between two change points. Defaults to 10.
        min_size (int, optional): The minimum size of a segment. Defaults to 70.
        win_width (int, optional): The width of the sliding window for the 'win' method. Defaults to 100.

    Returns:
        list[int]: A list of indices where the changes occur.
    """
    f_method = {  # Choose the method
        "binseg": rpt.Binseg,
        "dynp": rpt.Dynp,
        "win": rpt.Window,
    }
    params = {  # Set method parameters inc. model choice
        "model": cost,
        "jump": jump,
        "min_size": min_size,
    }
    if method == "win":
        params["width"] = win_width

    # Find the changes
    algo = f_method[method](**params).fit(arr)
    return algo.predict(n_bkps=n_bkps)[:-1]


def filefinder(dir: Path, start: int, stop: int) -> Iterator[Path]:
    """Returns next sdf file."""
    yield from (dir / f"{i:04d}.sdf" for i in range(start, stop + 1))


def _load(
    file: Path,
    dt: float,
) -> Step:
    """Load a single .sdf file into Step instance."""
    return Step(
        tstamp=float(file.stem) * dt,
        sdf=load_sdf_verified(file),
    )


# %% Acutally load the data

load = partial(_load, dt=deck.output.dt_snapshot)
files = filefinder(DATA_PATH, FSTART, FSTOP)
SDFs: list[Step]
with Pool(N_THREADS) as pool:
    SDFs = list(tqdm(pool.imap(load, files), total=n_files, desc="Loading SDFs"))


# %% Actually detect the changes


def _changes(
    step: Step, qtys: list[str], changes_kw: dict[str, str | int]
) -> list[Step]:
    changes: dict[str, npt.NDArray[np.float64]] = {}
    for qty in qtys:
        change_points = detect_changes(
            arr=np.median(step.get_qty(qty), axis=1),
            **changes_kw,
        )
        changes[qty] = np.array(change_points)
    return Step(
        tstamp=step.tstamp,
        sdf=step.sdf,
        changes=changes,
    )


changes_kw = dict(
    method="win",
    cost="l2",
    n_bkps=1,
    jump=10,
    min_size=70,
    win_width=100,
)
change_qtys = ["numberdensity"]
changes = partial(
    _changes,
    qtys=change_qtys,
    changes_kw=changes_kw,
)


with Pool(N_THREADS) as pool:
    SDFs = list(tqdm(pool.imap(changes, SDFs), total=n_files, desc="Detecting changes"))

# %% Plot the changes


def get_smoothed_arr(qty: str, steps: list[Step]):
    xl = steps[0].sdf.mid_grid.x.size
    yl = len(steps)
    arr = np.empty((xl, yl))
    for i, step in enumerate(steps):
        arr[:, i] = step.get_qty(qty).mean(axis=1)

    # return gaussian_filter(arr, sigma=arr.std())
    return arr


fig, ax = plt.subplots()


def plot_the_shock(
    ax: plt.Axes,
    ref_qty: str,
    steps: list[Step],
    shock: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
):
    v = get_smoothed_arr(ref_qty, steps)

    style = {}
    if ref_qty == "numberdensity":
        style["cmap"] = mpl.Cmaps.sequential
    else:
        style["cmap"] = mpl.Cmaps.diverging
        peak = np.abs(v).max()
        style["vmin"] = -peak
        style["vmax"] = peak

    im = ax.pcolormesh(
        x,
        time,
        v.T,
        **style,
        rasterized=True,
    )
    ax.plot(shock, time, ls="-", color=colours.green())

    return im


grid = SDFs[0].sdf.mid_grid * (1 / deck.constant.di)
time = np.array([step.tstamp for step in SDFs])

list_changes = [step.changes for step in SDFs]
keys = list(list_changes[0].keys())
changes = np.empty((len(SDFs), len(keys)), dtype=int)
for i, k in enumerate(keys):
    v = np.array([d[k] for d in list_changes]).astype(int)
    changes[:, i] = v.T
    ax.plot(grid.x[v], time, ls="--", color=colours.dark())
# plot minimum change point of all qtys for each time
str_start = grid.x[changes.min(axis=1)]

ref_qty = "numberdensity"
im = plot_the_shock(ax, ref_qty, SDFs, str_start, grid.x, time)

fig.colorbar(im, ax=ax, pad=0.01, label=ref_qty)
ax.set_xlabel("x $d_i$")
ax.set_ylabel("Time $s$")
ax.set_title(
    f"Use {changes_kw['method']}:{changes_kw['cost']} on {', '.join(change_qtys)}"
)

fig.tight_layout()
plt.show()

# %% Refine str_start. Remove outliers and large differences

line = lambda x, m, c: m * x + c
fit, cov = curve_fit(line, time, str_start)

fig, ax = plt.subplots()
ax.scatter(
    str_start,
    time,
    marker=".",
    s=5,
    color=colours.red(),
    label="Changes",
)
ax.plot(
    line(time, *fit),
    time,
    ls=":",
    color=colours.green(),
    label=f"Fit {fit[0]:.1f}x + {fit[1]:.1f}",
)

resids = str_start - line(time, *fit)
resid_sd = resids.std()
outlier_mask = np.abs(resids) > 1 * resid_sd

diff_thresh = 2  # d_i
str_out = str_start.copy()
str_out[outlier_mask] = np.nan
diff_mask = np.insert(np.abs(np.diff(str_out)) > diff_thresh, 0, False)

str_refine = np.interp(
    time,
    time[~outlier_mask & ~diff_mask],
    str_start[~outlier_mask & ~diff_mask],
)

ax.scatter(
    str_start[outlier_mask],
    time[outlier_mask],
    marker="o",
    s=25,
    edgecolor=colours.dark(),
    facecolor="none",
    label="Outliers",
)

ax.plot(
    str_start[~outlier_mask],
    time[~outlier_mask],
    ls="-.",
    color=colours.green(),
    label="No fit outliers",
    alpha=0.5,
)

ax.scatter(
    str_start[diff_mask],
    time[diff_mask],
    marker="D",
    s=25,
    edgecolor=colours.dark(),
    facecolor="none",
    label="Large diff",
)


ax.plot(
    str_refine,
    time,
    ls="-",
    color=colours.green(),
    label="Refined",
)

ax.legend()


hist_ax = ax.inset_axes([0.05, 0.05, 0.35, 0.35])
hist_ax.hist(
    resids,
    bins=25,
    edgecolor="k",
    facecolor="none",
    density=True,
)
dist_x = np.linspace(*hist_ax.get_xlim(), 100)
hist_ax.plot(
    dist_x,
    norm.pdf(dist_x, loc=resids.mean(), scale=resids.std()),
    color=colours.red(),
)
hist_ax.grid(False)
hist_ax.set_xticklabels([])
hist_ax.set_yticklabels([])


fig.tight_layout()
plt.show()

# %% Plot str_refine
fig, ax = plt.subplots()
im = plot_the_shock(
    ax,
    "mag.bz",
    SDFs,
    str_refine,
    grid.x,
    time,
)
fig.colorbar(im, ax=ax, pad=0.01, label=ref_qty)
ax.set_xlabel("x $d_i$"),
ax.set_ylabel("Time $s$")
ax.set_title("Refined shock location")

fig.tight_layout()
plt.show()

# %% Really simple method

fig, ax = plt.subplots()

v = get_smoothed_arr("numberdensity", SDFs)
mask = v > (v.max() / np.e)
get_first_true = lambda arr: np.nonzero(arr)[0][0]
thresh_shock = np.asarray(
    [grid.x[get_first_true(mask[:, i])] for i in range(mask.shape[1])]
)

ref_qty = "numberdensity"
im = plot_the_shock(ax, ref_qty, SDFs, thresh_shock, grid.x, time)
fig.colorbar(im, ax=ax, pad=0.01, label=ref_qty)

ax.set_xlabel("x $d_i$")
ax.set_ylabel("Time $s$")
ax.set_title("Shock based on first x where density > max(density)/e")

fig.tight_layout()
plt.show()

# %% animate the shock 1d

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.set_xlabel("x $d_i$")
ax.set_ylabel("Numberdensity $cm^{-3}$")

nds = np.asarray([np.median(step.sdf.numberdensity, axis=1) / 10**3 for step in SDFs])
ax.set_ylim(0, nds.max())

im = ax.plot(
    grid.x,
    nds[0, :],
    color="k",
    ls="-",
)[0]
t = ax.text(
    10,
    0.9 * nds.max(),
    f"t = {time[0]:6.1f} $s$",
)
fig.tight_layout()


def animate(frame: int):
    im.set_ydata(nds[frame, :])
    t.set_text(f"t = {time[frame]:6.1f} $s$")
    return im, t


ani = FuncAnimation(fig, animate, interval=100, frames=range(1, len(nds)), blit=True)
ani.save("nd.mp4", dpi=300, fps=15)
