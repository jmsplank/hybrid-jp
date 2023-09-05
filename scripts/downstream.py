"""Follows from find_the_shock.py

Use the max(nd)/e method to find the shock location. Then look at the downstream.
"""
# %%
# Imports
from functools import partial
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from phdhelper import mpl
from phdhelper.colours import sim as colours
from typing import Callable
from epoch_cheats import validate_deck, evaluate_deck
from pathlib import Path
from multiprocessing import Pool, set_start_method
from hybrid_jp.sdf_files import SDF, load_sdf_verified, filefinder
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import matplotlib

# %%
# CONSTS
mpl.format()
set_start_method("fork")

DATA_DIR = Path().resolve().parent / "U6T40"
FSTART, FSTOP = 20, 200

n_files = FSTOP - FSTART + 1
deck = validate_deck(evaluate_deck(DATA_DIR / "input.deck"))

# %%
# Load data in parallel
files = filefinder(DATA_DIR, FSTART, FSTOP)
SDFs: list[SDF] = []

with Pool() as pool:
    SDFs = list(tqdm(pool.imap(load_sdf_verified, files), total=n_files))

# %%
# Append a timestamp to each interval

dt = deck.output.dt_snapshot
for n, sdf in zip(range(FSTART, FSTOP + 1), SDFs):
    sdf.tstamp = n * dt

# %%
# Find the shock location
grid = SDFs[0].mid_grid * (1 / deck.constant.di)
time = np.array([sdf.tstamp for sdf in SDFs])
nd = np.asarray([np.median(sdf.numberdensity, axis=1) for sdf in SDFs]).T / (100**3)

# Find first true for each row in nd_mask
nd_mask = nd > nd.max() / np.e
shock_i = np.asarray(np.argmax(nd_mask, axis=0))
shock_x = grid.x[shock_i]

# %%
# Quick plot
fig, ax = plt.subplots()
ax.pcolormesh(
    grid.x,
    time,
    nd.T,
)
ax.plot(shock_x, time, color="k", lw=1)
plt.show()

# %%

dist_either_side = np.asarray([[s, abs(s - grid.x.size)] for s in shock_i])
widths = dist_either_side.max(axis=0)

d_shock = np.empty((sum(widths), time.size))
d_shock[:] = np.nan

for i, s in enumerate(shock_i):
    d_shock[widths[0] - s : (widths[0] - s) + grid.x.size, i] = nd[:, i]

dx = grid.x[1] - grid.x[0]
dist_x = np.arange(-widths[0], widths[1]) * dx

# %%
axs: list[plt.Axes]
fig, _axs = plt.subplots(
    2,
    2,
    gridspec_kw=dict(height_ratios=[3, 1], width_ratios=[98, 2]),
)
axs = np.asarray(_axs).flatten()
axs[0].sharex(axs[2])
axs[3].axis("off")

im = axs[0].pcolormesh(
    dist_x,
    time,
    d_shock.T,
)
fig.colorbar(im, cax=axs[1], label="nd $cm^{-3}$")


axs[2].plot(dist_x, np.nanmedian(d_shock, axis=1), color="k")


axs[0].set_ylabel("Time $s$")
axs[2].set_ylabel("Median nd $cm^{-3}$")
axs[2].set_xlabel("Distance downstream of shock $d_i$")

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()

# %%
# Split the data into chunks

chunk_x_ideal = 30  # di
chunk_i = int(chunk_x_ideal / dx)
chunk_x = chunk_i * dx
n_chunks = int(dist_x.shape[0] // chunk_i)
missing = widths % chunk_i
missing[1] = sum(widths) - missing[1]
missing = slice(*missing)

# Chop off from start since downstream is more interesting
chunk_dist_x = d_shock[missing, :]
x_crop = dist_x[missing]

valid_chunks = np.zeros((n_chunks, time.size), dtype=bool)
for i in range(time.size):
    valid = [
        np.isfinite(chunk_dist_x[j * chunk_i : (j + 1) * chunk_i, i]).all()
        for j in range(n_chunks)
    ]
    valid_chunks[:, i] = np.asarray(valid)

starts = widths[0] - shock_i - missing.start
offsets = valid_chunks.copy().astype(np.int32)
offsets = (offsets * offsets.cumsum(axis=0) - 1) * chunk_i
offsets -= (np.argmax(valid_chunks, axis=0) * chunk_i) - starts
first_chunk = np.argmax(valid_chunks, axis=0) * chunk_i

# ######################################################################################
start_i = first_chunk - starts
len_all_chunks = (valid_chunks.sum(axis=0) * chunk_i) - 1
# ######################################################################################

fig, ax = plt.subplots()
ax.pcolormesh(
    x_crop,
    time,
    chunk_dist_x.T,
    cmap=colours.dark.cmap(),
)
ax.pcolormesh(x_crop, time, valid_chunks.repeat(chunk_i, axis=0).T, alpha=0.5)


dist_ax = ax.inset_axes([0.06, 0.72, 0.20, 0.20])
dist_ax.hist(
    np.repeat(x_crop[::chunk_i], valid_chunks.sum(axis=1)),
    bins=n_chunks,
    histtype="step",
)
dist_ax.axvline(0, color="k", ls="--")
dist_ax.xaxis.set_major_locator(MultipleLocator(100))
dist_ax.yaxis.set_major_locator(MultipleLocator(50))
dist_ax.tick_params(axis="both", which="major", labelsize=7)
dist_ax.set_title("# chunks", fontsize=8)
dist_ax.set_xlabel("x $d_i$", fontsize=8)
dist_ax.grid(False)


ax.set_ylabel("Time $s$")
ax.set_xlabel("Distance downstream of shock $d_i$")
ax.set_title(f"Data split into chunks of {chunk_x:.2f} $d_i$ ({chunk_i} points)")


ax.grid(False)
ax.grid(True, axis="x", which="both", alpha=0.3)
ax.xaxis.set_minor_locator(MultipleLocator(chunk_x))
ax.xaxis.set_major_locator(MultipleLocator(5 * chunk_x))
fig.tight_layout()
plt.show()

# %%
# Power spectrum for each chunk


def hann_2d(nx: int, ny: int) -> npt.NDArray[np.float64]:
    """Create a 2d hanning window.

    https://stackoverflow.com/a/65948798

    Example:
        >>> h2d = hann_2d(66, 160)
        >>> X, Y = np.meshgrid(np.arange(66), np.arange(160))
        >>> fig = plt.figure()
        >>> ax = plt.axes(projection="3d")
        >>> ax.contour3D(X, Y, h2d.T, 50)
        >>> plt.show()
    """
    hann_1d = [np.hanning(i) for i in (nx, ny)]
    return np.sqrt(np.outer(*hann_1d))


def power_xy(
    arr_xy: npt.NDArray[np.float64], dx: float, dy: float
) -> npt.NDArray[np.float64]:
    """Calculate the power spectrum of a 2d array.

    Note:
        https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    """
    freq_x = np.fft.fftfreq(arr_xy.shape[0], dx)
    freq_y = np.fft.fftfreq(arr_xy.shape[1], dy)

    arr_xy = (arr_xy - arr_xy.mean()) * hann_2d(*arr_xy.shape)
    kxky = np.fft.fft2(arr_xy)
    Pxy = np.abs(kxky) ** 2 / (dx * dy)

    kx_mask = np.expand_dims(freq_x > 0, axis=1)
    ky_mask = np.expand_dims(freq_y > 0, axis=0)
    freq_x = freq_x[freq_x > 0]
    freq_y = freq_y[freq_y > 0]
    return freq_x, freq_y, Pxy[kx_mask & ky_mask].reshape(freq_x.size, freq_y.size)


mag_z = np.asarray([sdf.mag.bz for sdf in SDFs])


# Crop to size of chunk #24
def get_chunk_at_t(
    chunk_n: int,
    t_n: int,
    arr: npt.NDArray[np.float64],
    chunk_w: int,
    valid_chunks: npt.NDArray[np.float64],
    chunk_start: npt.NDArray[np.float64],
) -> np.ndarray:
    if not valid_chunks[chunk_n, t_n]:
        raise Exception("Chunk is not valid")

    start = chunk_start[t_n] + valid_chunks[:chunk_n, t_n].sum() * chunk_w
    end = start + chunk_w
    return arr[t_n, start:end, :]


# fig, ax = plt.subplots()
# tmp = offsets.copy()
# tmp[~valid_chunks] = 0
# im = ax.pcolormesh(tmp.T, cmap=mpl.Cmaps.diverging, vmin=-tmp.max(), vmax=tmp.max())
# fig.colorbar(im)

chunk_test = get_chunk_at_t(
    chunk_n=valid_chunks.shape[0] // 2,
    t_n=np.argmin(valid_chunks[valid_chunks.shape[0] // 2, :]),
    chunk_w=chunk_i,
    arr=mag_z,
    valid_chunks=valid_chunks,
    chunk_start=start_i,
)
dy = grid.y[1] - grid.y[0]
kx, ky, Pxy = power_xy(chunk_test, dx, dy)

fig, ax = plt.subplots()

ax.pcolormesh(kx, ky, Pxy.T, cmap=mpl.Cmaps.sequential, norm=LogNorm())
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_aspect("equal")
ax.set_xlabel("$k_x$ $d_i^{-1}$")
ax.set_ylabel("$k_y$ $d_i^{-1}$")
ax.set_title("Power for one chunk at one time")

ax.grid(False)


fig.tight_layout()
plt.show()

# %%
# Convert to polar coordinates

r_max = np.sqrt(kx.max() ** 2 + ky.max() ** 2)
r_min = np.sqrt(kx.min() ** 2 + ky.min() ** 2)
n_r_bins = 20
KX, KY = np.meshgrid(kx, ky)
r = np.sqrt(KX**2 + KY**2).T
r_bin_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_r_bins + 1)


def get_mean_in_bin(
    edges: npt.NDArray[np.float64],
    arr: npt.NDArray[np.float64],
    r_values: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    Pr = np.zeros(edges.size - 1)
    for i, bin in enumerate(edges[:-1]):
        mask = (r_values >= bin) & (r_values < edges[i + 1])
        if mask.sum() > 0:
            Pr[i] = arr[mask].mean()
        else:
            Pr[i] = np.nan
    return Pr


Pr = get_mean_in_bin(r_bin_edges, Pxy, r)
fig, ax = plt.subplots()
ax.stairs(Pr, r_bin_edges, color="k", zorder=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$k$ $d_i^{-1}$")
ax.set_ylabel("$P(k)$")
ax.set_title("Integration over $k_x$ and $k_y$")


fig.tight_layout()
plt.show()

# %% Mean over all times for one chunk

chunk_number = 5
valid_time_i_for_chunk = np.nonzero(valid_chunks[chunk_number, :])[0]
chunk_data = partial(
    get_chunk_at_t,
    chunk_n=chunk_number,
    chunk_w=chunk_i,
    arr=mag_z,
    chunk_start=start_i,
    valid_chunks=valid_chunks,
)

chunk_all_times = np.empty((n_r_bins, valid_time_i_for_chunk.size))
for i in range(valid_time_i_for_chunk.size):
    chunk = chunk_data(t_n=valid_time_i_for_chunk[i])
    _, _, Pxy = power_xy(chunk, dx, dy)
    chunk_all_times[:, i] = get_mean_in_bin(r_bin_edges, Pxy, r)

chunk_avg = chunk_all_times.mean(axis=1)

fig, ax = plt.subplots()
ax.stairs(chunk_avg, r_bin_edges, color="k", zorder=2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$k$ $d_i^{-1}$")
ax.set_ylabel("$P(k)$")
ax.set_title("Mean over all times for one chunk")

fig.tight_layout()
plt.show()

# %% Mean power for all chunks
chunks = np.arange(n_chunks)

f_chunk = partial(
    get_chunk_at_t,
    chunk_w=chunk_i,
    arr=mag_z,
    chunk_start=start_i,
    valid_chunks=valid_chunks,
)  # f(t_n:int, chunk_n: int) -> arr_xy

f_power = partial(
    power_xy,
    dx=dx,
    dy=dy,
)  # f(arr_xy:npt.NDArray[np.float64]) -> (freq_x, freq_y, Pxy)

f_mean_in_bin = partial(
    get_mean_in_bin,
    edges=r_bin_edges,
    r_values=r,
)  # f(arr:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]


def power_in_chunk(
    chunk: int,
    f_chunk_data: Callable[[int, int], npt.NDArray[np.float64]],
    f_power: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    f_mean_in_bin: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    n_r_bins: int,
    valid_chunks: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    valid_times = np.nonzero(valid_chunks[chunk, :])[0]
    chunk_all_times = np.empty((n_r_bins, valid_times.size))
    for i in range(valid_times.size):
        chunk_data = f_chunk_data(chunk_n=chunk, t_n=valid_times[i])
        _, _, Pxy = f_power(chunk_data)
        chunk_all_times[:, i] = f_mean_in_bin(arr=Pxy)

    return chunk_all_times.mean(axis=1)


f_chunk_power = partial(
    power_in_chunk,
    f_chunk_data=f_chunk,
    f_power=f_power,
    f_mean_in_bin=f_mean_in_bin,
    n_r_bins=n_r_bins,
    valid_chunks=valid_chunks,
)

out = []
for chunk in tqdm(chunks):
    out.append(f_chunk_power(chunk))

# %%

fig = plt.figure()
gs = GridSpec(2, 2, width_ratios=[98, 2])
ax = fig.add_subplot(gs[:, 0])
cbar_post = fig.add_subplot(gs[0, 1])
cbar_pre = fig.add_subplot(gs[1, 1])

shock_chunk = np.argmin(np.abs(x_crop)) // chunk_i
vmax = max(shock_chunk, len(out) - shock_chunk)
l_after = len(out) - shock_chunk

for i in range(len(out)):
    style = {}
    # if i < shock_chunk:
    #     style["color"] = colours.blue.cmap(n=shock_chunk * 2)(i + (shock_chunk // 2))
    # else:
    #     style["color"] = colours.red.cmap(n=l_after * 2)(
    #         (i - shock_chunk) + l_after // 2
    #     )
    if i < shock_chunk:
        style["color"] = colours.blue.cmap(n=shock_chunk)(i)
    else:
        style["color"] = colours.red.cmap(n=l_after)(i - shock_chunk)
    ax.plot(r_bin_edges[:-1], out[i], ls="-", **style)

sm_pre = matplotlib.cm.ScalarMappable(cmap=colours.blue.cmap())
sm_post = matplotlib.cm.ScalarMappable(cmap=colours.red.cmap())

fig.colorbar(sm_pre, cax=cbar_pre, label="Before shock")
fig.colorbar(sm_post, cax=cbar_post, label="After shock")
cbar_post.set_yticks([1], labels=["end"])
cbar_pre.set_yticks([0, 1], labels=["start", "shock"])


ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("$k$ $d_i^{-1}$")
ax.set_ylabel("$P(k)$")
ax.set_title(f"Windows of {chunk_x:.2f} $d_i$ ({chunk_i} points)")

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()

# %%
# Slopes

from scipy.optimize import curve_fit
from hybrid_jp.arrays import interpolate_to_midpoints


def line(x, m, c):
    return m * x + c


def compute_slope(
    arr_P: npt.NDArray[np.float64],
    arr_x: npt.NDArray[np.float64],
    start_di: float = 0.1,
    end_di: float = 1.0,
) -> tuple[float, float]:
    y = arr_P[(arr_x >= start_di) & (arr_x <= end_di)]
    x = arr_x[(arr_x >= start_di) & (arr_x <= end_di)]

    popt, pcov = curve_fit(line, np.log10(x), np.log10(y))
    return popt[0], np.sqrt(np.diag(pcov))[0]


slopes = []
slopes_err = []

x = interpolate_to_midpoints(r_bin_edges, 2)

for i in range(len(out)):
    fit = compute_slope(out[i], x)
    slopes.append(fit[0])
    slopes_err.append(fit[1])

slopes = np.asarray(slopes)
slopes_err = np.asarray(slopes_err)
dists = np.linspace(dist_x[0], dist_x[-1], len(out))

fig, ax = plt.subplots()

ax.errorbar(dists, slopes, yerr=slopes_err, fmt="o")
ax.axhline(0, color=colours.dark.c800, ls="--")
ax.axvline(0, color=colours.dark.c800, ls="--")
fig.tight_layout()
plt.show()
