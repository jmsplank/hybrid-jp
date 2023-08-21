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

from hybrid_jp.sdf_files import SDF, load_sdf_verified

mpl.format()

# %% Constants
DATA_PATH = Path().resolve().parent / "U6T40"
FSTART, FSTOP = 20, 200

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


# %% Load all SDFs
SDFs = []
files = filefinder(DATA_PATH, FSTART, FSTOP)
for f in tqdm(files):
    step = Step(
        tstamp=float(f.stem) * deck.output.dt_snapshot,
        sdf=load_sdf_verified(f),
    )
    SDFs.append(step)


# %% Animation (Slow!)
# animate_sdfs(SDFs, Path("test.mp4"))

# %% Detect Changes
SDFs_changes: list[Step] = []
for i, sdf in tqdm(enumerate(SDFs)):
    changes = detect_changes(
        ChangePointMethod.WIN,
        ChangePointCost.L1,
        sdf.all_params_arr(),
        jump=10,
        min_size=70,
        win_width=100,
    )
    SDFs_changes.append(Step(tstamp=sdf.tstamp, sdf=sdf.sdf, changes=changes))

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

skip = 1
n_steps = len(SDFs_changes) // skip

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
shunt_factor = 0.6
ax.plot(shock_i, np.arange(n_steps) * shunt_factor, color=colours.sim.red())

for i, sdf in enumerate(SDFs_changes[::skip]):
    v = gaussian_filter1d(sdf.sdf.numberdensity.mean(axis=1), 1).astype(np.float64)
    v = (v - v.mean()) / v.std()
    ax.plot(
        grid.x,
        v + shunt_factor * i,
        color="k",
        ls="-",
    )


ax.set_yticks([])
ax.set_xlabel("x $d_i$")
ax.set_ylabel(r"$\leftarrow$ Time $t$")
ax.grid(False)
plt.tight_layout()
plt.show()

# %%

axs: list[plt.Axes]
fsize_key = 8
fig, axs = plt.subplots(1, 3, figsize=(fsize_key, fsize_key / 3))  # type: ignore

times = np.asarray([s.tstamp for s in SDFs_changes])
axs[0].scatter(times, shock_i, color=colours.sim.dark(), marker="+")  # type: ignore

line = lambda x, m, c: m * x + c

fit = curve_fit(line, times, shock_i)

axs[0].plot(
    times,
    line(times, *fit[0]),
    color=colours.sim.red(),
)

# resid
