# %% Imports
import time
from audioop import avgpp
from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize
from numpy import typing as npt
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy import optimize
from tqdm import tqdm

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %% Load in data
START, END = 20, 200
set_start_method("fork")
mpl.format()
data_folder = Path().resolve().parent / "U6T40"
deck = hja.load_deck(data_dir=data_folder)
SDFs, fpaths = hja.load_sdfs_para(
    sdf_dir=data_folder,
    dt=deck.output.dt_snapshot,
    threads=7,
    start=START,
    stop=END,
)
cs = hja.CenteredShock(SDFs, deck)

# %% Split into chunks
N_CHUNKS = 10
cs.n_chunks = N_CHUNKS
shock_chunk = cs.downstream_start_chunk


def get_bx(sdf: hj.sdf_files.SDF) -> npt.NDArray[np.float64]:
    return sdf.mag.bx


t = int(np.argmax(cs.valid_chunks[shock_chunk, :]))
qty, slc = cs.get_qty_in_frame(get_bx, shock_chunk, t)
plt.pcolormesh(
    cs.grid_km.x[slice(*slc)] - cs.grid_km.x[cs.get_x_offset_for_frame(shock_chunk, t)],
    cs.grid_km.y,
    qty.T,
)
plt.tight_layout()
plt.show()


# %%
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


arrf = npt.NDArray[np.float64]


def power_xy(
    arr: arrf,
    dx: float,
    dy: float,
) -> tuple[arrf, arrf, arrf]:
    """Get power spectrum of an array f(x, y).

    Args:
        arr (npt.NDArray[np.float64]): Array of shape (nx, ny)
        dx (float): Spacing between x points
        dy (float): Spacing between y points

    Returns:
        Pxy (npt.NDArray[np.float64]): Power spectrum of arr
        fx (npt.NDArray[np.float64]): x frequencies
        fy (npt.NDArray[np.float64]): y frequencies
    """
    fx = np.fft.fftfreq(arr.shape[0], dx)
    fy = np.fft.fftfreq(arr.shape[1], dy)

    # Center the data and apply Hanning window
    arr = (arr - arr.mean()) * hann_2d(*arr.shape)
    kxky = np.fft.fft2(arr)
    Pxy = np.abs(kxky) ** 2 / (dx * dy)

    # Mask out the negative frequencies
    # expand_dims is needed to broadcast the masks over the correct dimensions in 2d
    # The x mask needs to cover all y so add an extra dimension at axis 1 (y)
    fx_mask = fx > 0
    fx_mask2 = np.expand_dims(fx_mask, axis=1)  # shape (nx, 1)
    # The y mask needs to cover all x so add an extra dimension at axis 0 (x)
    fy_mask = fy > 0
    fy_mask2 = np.expand_dims(fy_mask, axis=0)  # shape (1, ny)

    fx = fx[fx_mask]
    fy = fy[fy_mask]
    Pxy = Pxy[fx_mask2 & fy_mask2].reshape(fx.size, fy.size)

    return Pxy, fx, fy


qty_func = Callable[[hj.sdf_files.SDF], npt.NDArray[np.float64]]


def subdivide_repeat(
    subdivisons: int,
    arrxy: arrf,
    x: arrf,
    y: arrf,
) -> tuple[arrf, arrf, arrf]:
    """Subdivide an array and repeat the values.

    Args:
        subdivisons (int): Number of subdivisions in each dimension.
        arrxy (arrf): Array to subdivide.
        x (arrf): x values.
        y (arrf): y values.

    Returns:
        arrxy (arrf): Subdivided array.
        x (arrf): Subdivided x values.
        y (arrf): Subdivided y values.
    """
    arrxy = np.repeat(arrxy, subdivisons, axis=0)
    arrxy = np.repeat(arrxy, subdivisons, axis=1)
    x = np.repeat(x, subdivisons)
    y = np.repeat(y, subdivisons)
    return arrxy, x, y


def radial_power(
    Pxy: arrf,
    fx: arrf,
    fy: arrf,
    subdivisions: int,
    n_bins: int,
) -> tuple[arrf, arrf]:
    """Get the radial bins for a power spectrum.

    Args:

    Returns:
        Pr (npt.NDArray[np.float64]): Power in each radial bin.
        r_edges (npt.NDArray[np.float64]): Edges of the radial bins.
    """
    r_func = lambda a, b: np.sqrt(a**2 + b**2)
    r_max = np.log10(r_func(fx.max(), fy.max()))
    r_min = np.log10(r_func(fx.min(), fy.min()))

    # subdivide Pxy by dividing each cell into subdivisions^2 cells
    # ie if subdivisions = 2, each cell will be divided into 4, double in x and in y
    # the subdivided cells will be filled with the same value as the original cell
    Pxy = subdivide_repeat(subdivisions, Pxy, fx, fy)[0]
    kx_s = np.linspace(fx.min(), fx.max(), fx.size * subdivisions)
    ky_s = np.linspace(fy.min(), fy.max(), fy.size * subdivisions)

    KX, KY = np.meshgrid(kx_s, ky_s)
    R = r_func(KX, KY).T
    r_edges = np.logspace(r_min, r_max, n_bins + 1)

    Pr = np.zeros(n_bins)
    for i in range(n_bins):
        mask: npt.NDArray[np.bool_] = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if not mask.any():
            Pr[i] = np.nan
        else:
            Pr[i] = Pxy[mask].mean()
    return Pr, r_edges


def frame_power(
    cs: hja.CenteredShock,
    chunk: int,
    t_idx: int,
    subdivisions: int,
    n_bins: int,
    centres: bool = True,
):
    def _func(
        sdf: hj.sdf_files.SDF, qty: Literal["bx", "by", "bz"]
    ) -> npt.NDArray[np.float64]:
        return getattr(sdf.mag, qty)

    xfunc = partial(_func, qty="bx")
    yfunc = partial(_func, qty="by")
    zfunc = partial(_func, qty="bz")
    cfuncs = [xfunc, yfunc, zfunc]

    p_components = np.empty((n_bins, 3))
    edges = np.empty(n_bins)
    for i, component in enumerate(cfuncs):
        data = cs.get_qty_in_frame(component, chunk, t_idx)[0]
        pxy, kx, ky = power_xy(data, cs.dx, cs.dy)
        pr, r_edges = radial_power(pxy, kx, ky, subdivisions, n_bins)
        p_components[:, i] = pr
        edges = r_edges

    if centres:
        kr = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), n_bins)
    else:
        kr = edges
    return p_components.sum(axis=1), kr


# %%
bz_func = lambda x: x.mag.bz
bz = cs.get_qty_in_frame(bz_func, shock_chunk, t)[0]
pxy, kx, ky = power_xy(bz, cs.dx, cs.dy)
axs: list[plt.Axes]
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # type: ignore
axs[0].pcolormesh(kx, ky, pxy.T, norm=LogNorm())
axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].set_aspect("equal")

pxy4, kx4, ky4 = subdivide_repeat(4, pxy, kx, ky)
axs[1].pcolormesh(kx4, ky4, pxy4.T, norm=LogNorm())
axs[1].set_yscale("log")
axs[1].set_xscale("log")
axs[1].set_aspect("equal")
fig.tight_layout()
plt.show()

# %%
n_r_bins: int = 100
n_sub: int = 8
Pr4, kr = radial_power(pxy, kx, ky, n_sub, n_r_bins)
kr = np.logspace(np.log10(kr[0]), np.log10(kr[-1]), n_r_bins)
print(Pr4.shape, kr.shape, n_r_bins * n_sub)

fig, ax = plt.subplots()  # type: ignore
ax.plot(kr, Pr4, c="k", ls="-")
ax.set_xscale("log")
ax.set_yscale("log")

fig.tight_layout()
plt.show()

# %%
pr, kr = frame_power(cs, shock_chunk, t, n_sub, n_r_bins)
fig, ax = plt.subplots()
ax.loglog(kr, pr, c="k", ls="-")
fig.tight_layout()
plt.show()


# %%
vc = cs.valid_chunks
val: list[list[int]] = [list(np.nonzero(vc[i, :])[0]) for i in range(vc.shape[0])]


def get_power_para(
    cs: hja.CenteredShock,
    val: list[list[int]],
    n_r_bins: int,
    n_sub: int,
):
    assert cs.n_chunks is not None

    kr = np.empty(n_r_bins)
    avg_pwr = np.empty((n_r_bins, cs.n_chunks))

    start_time = time.time()
    for i, v in enumerate(val):
        out = np.empty((n_r_bins, len(v)))
        for j, t in tqdm(enumerate(v), total=len(v), desc=f"Chunk {i}/{cs.n_chunks-1}"):
            out[:, j], kr = frame_power(cs, i, t, n_sub, n_r_bins)
        avg_pwr[:, i] = out.mean(axis=1)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.1f}s")

    return avg_pwr.T, kr


avg_pwr, kr = get_power_para(cs, val, n_r_bins, n_sub)

# %%
fig, axs = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[97, 3]))  # type: ignore
ax = axs[0]
cax = axs[1]


segments = np.array([np.stack([kr] * cs.n_chunks, axis=0), avg_pwr])
segments = np.moveaxis(segments, [0, 1, 2], [2, 0, 1])
norm = Normalize(vmin=0, vmax=cs.n_chunks)
lc = LineCollection(
    segments,  # type: ignore
    array=np.arange(cs.n_chunks),
    norm=norm,
    cmap=mpl.Cmaps.isoluminant,
)  # type: ignore
ax.add_collection(lc)  # type: ignore
ax.loglog(*segments[shock_chunk].T, c=colours.red())
ax.set_xscale("log")
ax.set_yscale("log")

fig.colorbar(lc, cax=cax, label="Chunk #")
cax.axhline(shock_chunk, c=colours.red(), lw=2)

ax.set_xlabel("$k$ [$km^{-1}$]")
ax.set_ylabel("$P(k)$")

fig.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()


# %%
# Slopes
def powerlaw(x, amplitude, exponent, constant):
    return amplitude * x**exponent + constant


def line(x, m, c):
    return m * x + c


min_x = 2e-3
max_x = 2e-2


def fitline(arr, min_x, max_x, func):
    # arr.shape = (n_chunks, 2) where 2 is (x, y)
    mask = (arr[:, 0] >= min_x) & (arr[:, 0] <= max_x)
    arr = np.log10(arr[mask, :])
    fit, _ = optimize.curve_fit(func, arr[:, 0], arr[:, 1])
    return fit


fig, ax = plt.subplots()
slopes = np.empty((cs.n_chunks, 2))
for i, s in enumerate(segments):
    slopes[i, :] = fitline(s, min_x, max_x, line)
    ax.plot(*np.log10(s).T, c="k", ls="-", lw=0.5)
    xr = np.log10(np.linspace(min_x, max_x, 2))
    ax.plot(xr, line(xr, *slopes[i, :]), c="r", ls="--")

plt.show()

# %%
fig, ax = plt.subplots()

dists = (
    (np.arange(cs.n_chunks) - shock_chunk)
    * cs.dx
    * cs.chunk_i
    / (deck.constant.di * 1e-3)
)
ax.scatter(dists, slopes[:, 0], c="k", marker="x")  # type: ignore

ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel("Power law index")


fig.tight_layout()
plt.show()
