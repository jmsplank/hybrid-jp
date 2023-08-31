"""Follows from change_point_experiments.py.

Aim is to use the same change point serach methods and cost functions
for every timestep of simulation and see how consistent it is.
"""
# %% Imports
from enum import Enum, auto
from pathlib import Path
from typing import Iterator, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import ruptures as rpt
from epoch_cheats import evaluate_deck, validate_deck
from matplotlib.animation import FuncAnimation, writers
from matplotlib.ticker import MultipleLocator
from phdhelper import colours, mpl
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm
from multiprocessing import Pool, set_start_method
from functools import partial
from scipy.ndimage import gaussian_filter

from hybrid_jp.sdf_files import SDF, load_sdf_verified

mpl.format()
set_start_method("fork")  # Needed for multiprocessing in Jupyter

# %% Constants
DATA_PATH = Path().resolve().parent / "U6T40"
FSTART, FSTOP = 20, 200
N_THREADS = 7

n_files = FSTOP - FSTART + 1

# %% Load deck
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %% Define some data types


class ChangePointMethod(Enum):
    """Group of changepoint methods."""

    DYNP = auto()
    WIN = auto()
    BINSEG = auto()


class ChangePointCost(str, Enum):
    """Group of changepoint cost functions."""

    L1 = "l1"
    L2 = "l2"
    AR = "ar"
    NORM = "norm"


class Step(NamedTuple):
    """Time step of simulation."""

    tstamp: float
    sdf: SDF
    changes: list[int] = []

    def all_params_dict(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return all parameters as a dictionary."""
        rows: dict[str, npt.NDArray[np.float64]] = (
            dict(nd=self.sdf.numberdensity)
            | self.sdf.mag.all
            | self.sdf.elec.all
            | self.sdf.current.all
        )
        return rows

    def all_params_arr(self) -> npt.NDArray[np.float64]:
        """Return all parameters in a single array."""
        return np.column_stack(
            [block_normalise(v) for v in self.all_params_dict().values()]
        ).astype(np.float64)


# %% Functions


def detect_changes(
    method: ChangePointMethod,
    cost: ChangePointCost,
    arr: npt.NDArray[np.float64],
    jump: int = 10,
    min_size: int = 70,
    win_width: int = 100,
) -> list[int]:
    """Detect change points in array."""
    method_fns = {
        ChangePointMethod.BINSEG: rpt.Binseg,
        ChangePointMethod.DYNP: rpt.Dynp,
        ChangePointMethod.WIN: rpt.Window,
    }
    attrs = {
        "model": cost.value,
        "jump": jump,
        "min_size": min_size,
    }
    if method == ChangePointMethod.WIN:
        attrs["width"] = win_width
    algo = method_fns[method](**attrs).fit(arr)
    return algo.predict(n_bkps=2)[:-1]  # type: ignore


def animate_sdfs(frames: list[Step], save: Path):
    """Animate the SDFs."""
    fig, ax = plt.subplots()
    ax.set_xlabel("x $d_i$")
    ax.set_ylabel("Row #")
    ax.yaxis.set_major_locator(MultipleLocator(160))
    labels = list(frames[0].all_params_dict().keys())
    for i, lab in enumerate(labels):
        ax.text(
            15,
            i * 160 + 80,
            lab,
            color="k",
            horizontalalignment="center",
            verticalalignment="center",
        )

    arrs = [s.all_params_arr() for s in frames]
    minmax = [(a.min(), a.max()) for a in arrs]
    vmin, vmax = min([m[0] for m in minmax]), max([m[1] for m in minmax])
    x = frames[0].sdf.mid_grid.x / deck.constant.di

    im = ax.pcolormesh(
        x,
        np.arange(arrs[0].shape[1]),
        arrs[0].T,
        cmap=mpl.Cmaps.sequential,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, pad=0, label="Normalised by parameter")
    l1, l2 = None, None
    if frames[0].changes:
        l1 = ax.axvline(x[frames[0].changes[0]], color="k")
        l2 = ax.axvline(x[frames[0].changes[1]], color="k")

    def update(frame):
        im.set_array(arrs[frame].T)
        ax.set_title(f"t={frames[frame].tstamp:.2f}")
        if frames[frame].changes and l1 and l2:
            l1.set_xdata([x[frames[frame].changes[0]]])
            l2.set_xdata([x[frames[frame].changes[1]]])
        return [im, l1, l2]

    fig.tight_layout()

    ani = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    Writer = writers["ffmpeg"]
    ffmpeg_writer = Writer(fps=7, bitrate=1800)  # type: ignore
    ani.save(str(save), dpi=300, writer=ffmpeg_writer)


def block_normalise(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalise the array between 0 and 1."""
    return (arr - arr.min()) / (arr.max() - arr.min())


def filefinder(dir_path: Path, start: int, stop: int) -> Iterator[Path]:
    """Iterate over files and return next file (timestep) in simulation."""
    yield from (dir_path / f"{i:04d}.sdf" for i in range(start, stop + 1))


def _thread_loader(sdf_path: Path, dt: float) -> Step:
    """Multiprocessing wrapper for loading SDFs."""
    return Step(
        tstamp=float(sdf_path.stem) * dt,
        sdf=load_sdf_verified(sdf_path),
    )


def thread_changes(sdf: Step) -> Step:
    """Multiprocessing wrapper for detecting changes."""
    changes = detect_changes(
        ChangePointMethod.WIN,
        ChangePointCost.L1,
        sdf.all_params_arr(),
        jump=10,
        min_size=70,
        win_width=100,
    )
    return Step(tstamp=sdf.tstamp, sdf=sdf.sdf, changes=changes)


def plot_stacked_lines(
    ax: plt.Axes,
    y_offset: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    sdfs: list[Step],
):
    """Stacked line plot of numberdensity for a list of SDFs."""
    for i, sdf in enumerate(sdfs):
        v = gaussian_filter1d(sdf.sdf.numberdensity.mean(axis=1), 1).astype(np.float64)
        v = (v - v.mean()) / v.std()  # center and scale to unit variance
        ax.plot(
            x,
            v + y_offset[i],
            color="k",
            ls="-",
        )


def smoothed_qty(
    qty_name: str,
    steps: list[Step],
) -> npt.NDArray[np.float64]:
    """Median of qty over y axis, smoothed with a gaussian filter."""
    yl = len(steps)
    xl = len(steps[0].sdf.mid_grid.x)
    name = qty_name.split(".")
    v = np.empty((xl, yl))
    for i, sdf in enumerate(steps):
        _y = getattr(sdf.sdf, name[0])
        if len(name) > 1:
            y = getattr(_y, name[1])
        else:
            y = _y

        y = np.median(y, axis=1)
        v[:, i] = y
    v = v / (v.std() ** 2)
    return gaussian_filter(v, 1)
    return v


def fmt_label(s: str) -> str:
    """e.g. 'current.jx' -> 'Current, $J_x$'."""
    comp = s.split(".")
    if len(comp) == 1:
        return s
    name = comp[0].title()
    var = comp[1]
    var = f"{var[0].upper()}_{var[1:]}"
    return rf"{name}, $\frac{{{var}-|{var}|}}{{var({var})}}$"


# %% Load all SDFs, threaded
SDFs: list[Step] = []
files = filefinder(DATA_PATH, FSTART, FSTOP)

# Partial application of thread_loader with timestep
thread_loader = partial(_thread_loader, dt=deck.output.dt_snapshot)

# Load all SDFs in parallel
with Pool(N_THREADS) as pool:
    SDFs = list(tqdm(pool.imap(thread_loader, files), total=n_files))


# %% Animation (Slow!)
# animate_sdfs(SDFs, Path("test.mp4"))

# %% Detect Changes, threaded
SDFs_changes: list[Step] = []

# Detect changes in parallel
with Pool(N_THREADS) as pool:
    SDFs_changes = list(tqdm(pool.imap(thread_changes, SDFs), total=len(SDFs)))

# %% Animation
# animate_sdfs(SDFs_changes, Path("frames_with_changes.mp4"))

# %% Plot changes
fig, ax = plt.subplots()

x_di = SDFs_changes[0].sdf.mid_grid.x / deck.constant.di
# Plot the change points as black +
for i, sdf in enumerate(SDFs_changes):
    ax.scatter(
        [sdf.tstamp] * len(sdf.changes),
        x_di[sdf.changes],
        color="k",
        marker="+",  # type: ignore
    )

# Plot the peak numberdensity as red line
t = [s.tstamp for s in SDFs_changes]
x = [x_di[np.argmax(s.sdf.numberdensity.max(axis=1))] for s in SDFs_changes]
ax.plot(
    t,
    x,
    color=colours.sim.red(),
)

ax.set_xlabel("Time $t$")
ax.set_ylabel("x $d_i$")

plt.tight_layout()
plt.show()

# %% Components
fig, ax = plt.subplots()
step = SDFs_changes[120]
grid = step.sdf.mid_grid * (1 / deck.constant.di)
n_steps = len(SDFs_changes)

# Get x-coord of closest change point to peak numberdensity
shock_i = [
    grid.x[
        s.changes[
            np.argmin(
                np.abs(
                    np.asarray(s.changes) - s.sdf.numberdensity.mean(axis=1).argmax()
                )
            )
        ]
    ]
    for s in SDFs_changes
]

# y offset (aesthetic)
shunt_factor = 0.6
_y_steps = np.arange(n_steps) * shunt_factor

ax.plot(shock_i, _y_steps, color=colours.sim.red())
plot_stacked_lines(ax, _y_steps, grid.x, SDFs_changes)

ax.set_yticks([])
ax.set_xlabel("x, $d_i$")
ax.set_ylabel(r"Time, $t\quad\rightarrow$")
ax.grid(False)
plt.tight_layout()
plt.show()

# %% Fit a line to the changes and do some pruning of outliers

axs: list[plt.Axes]
fsize_key = 8
fig, axs = plt.subplots(
    2,
    2,
    figsize=[fsize_key] * 2,
)  # type: ignore
axs = np.asarray(axs).flatten()
axs[-1].axis("off")

times = np.asarray([s.tstamp for s in SDFs_changes])
axs[0].scatter(times, shock_i, color=colours.sim.dark(), marker="+")  # type: ignore

line = lambda x, m, c: m * x + c

fit = curve_fit(line, times, shock_i)


axs[0].plot(
    times,
    line(times, *fit[0]),
    color=colours.sim.red(),
)

# residuals
resids = shock_i - line(times, *fit[0])
std_r = np.std(resids)

axs[1].scatter(times, resids, marker="+")
style = dict(color=colours.sim.red(), ls="--")
axs[1].axhline(0, color=colours.sim.red(), ls="--")
axs[1].axhline(std_r, color=colours.sim.red(), ls=":")
axs[1].axhline(-std_r, color=colours.sim.red(), ls=":")


resids_mask = np.logical_and(-std_r < resids, resids < std_r)
n_resids_mask = np.logical_not(resids_mask)
axs[1].scatter(
    times[n_resids_mask],
    resids[n_resids_mask],
    marker="o",
    facecolor="none",
    edgecolor=colours.sim.red(),
    s=50,
)

f_shock_i = np.array(shock_i.copy())[resids_mask]
f_times = times.copy()[resids_mask]

i_shock_i = np.interp(times, f_times, f_shock_i)

axs[2].scatter(times, i_shock_i, color=colours.sim.dark(), marker="+")  # type: ignore

fit = curve_fit(line, times, i_shock_i)
axs[2].plot(
    times,
    line(times, *fit[0]),
    color=colours.sim.red(),
)

axs[0].set_title("Original")
axs[1].set_title("Residuals")
axs[2].set_title("Interpolated")

fig.tight_layout()
plt.show()


# %% Test more pruning on dummy data,
# this time remove also points where the change is too large

# Dummy array containing large changes and also NaNs
a = np.asarray([2, 4, 8, 9, np.nan, 10, 12, 14, np.nan, 18, 20, 25, 28, 30, 32, 36])
b = np.diff(a)


# plot a vertical line the length of b[i] at np.arange(1,len(b))
plt.vlines(np.arange(1, len(a)), a[1:], a[1:] - b)
plt.scatter(np.arange(len(a)), a, marker="+", color="k")

plt.grid(visible=True, which="minor", alpha=0.2)

mask = b > 2
mask = np.insert(mask, 0, False)
c = a.copy()
c[mask] = np.nan
d = c[np.isfinite(c)]
t = np.arange(len(c))[np.isfinite(c)]
plt.plot(t, d, color="r", ls="--")
plt.scatter(t, d, marker="o", facecolor="none", edgecolor="r", s=50)

# %% Apply method to shock position
# times
shock_nan = np.array(shock_i.copy())
shock_nan[~resids_mask] = np.nan
shock_d = np.diff(shock_nan)


fig, ax = plt.subplots()

ax.vlines(
    times[1:],
    shock_nan[1:],
    shock_nan[1:] - shock_d,
)
ax.scatter(times, shock_nan, marker=".", color="k", s=5)

mask = np.abs(shock_d) > 3  # d_i
mask = np.insert(mask, 0, False)
shock_morenan = shock_nan.copy()
shock_morenan[mask] = np.nan
shock_reduce = shock_morenan[np.isfinite(shock_morenan)]
times_reduce = times[np.isfinite(shock_morenan)]

shock = np.interp(times, times_reduce, shock_reduce)

ax.plot(times_reduce, shock_reduce, color="r", ls="--")
ax.scatter(
    times,
    shock,
    marker="o",
    facecolor="none",
    edgecolor="r",
    s=20,
)

fig.tight_layout()
plt.show()

# %% Plot the new shock position on a colourplot of qty_s

fig, ax = plt.subplots()

ax.plot(shock, times, color=colours.sim.dark())

# numberdensity | mag.bx,by,bz | elec.ex,ey,ez | current.jx,jy,jz
qty_s = "numberdensity"

qty = smoothed_qty(qty_s, SDFs_changes)
vmin, vmax = qty.min(), qty.max()
rng = (min(vmin, -vmax), max(-vmin, vmax))
style = {}
if qty_s == "numberdensity":
    style["cmap"] = mpl.Cmaps.sequential
else:
    style["cmap"] = mpl.Cmaps.diverging
    style["vmin"] = rng[0]
    style["vmax"] = rng[1]
im = ax.pcolormesh(grid.x, times, qty.T, **style)


fig.colorbar(im, ax=ax, pad=0.01, label=fmt_label(qty_s))

ax.set_xlabel("x, $d_i$")
ax.set_ylabel(r"Time, $s$")

plt.tight_layout()
plt.show()
