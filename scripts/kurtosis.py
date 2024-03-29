# %%
"""Determine the kurtosis for the data split into chunks.
"""
from functools import partial
from itertools import product
from multiprocessing import set_start_method
from os import environ
from time import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from dotenv import dotenv_values
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy.optimize import curve_fit  # type: ignore

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
# Load constants
set_start_method("fork")
mpl.format()
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk

# %%
# Load data
deck = hja.load_deck(cfg.data_dir)
SDFs, fpaths = hja.load_sdfs_para(
    sdf_dir=cfg.data_dir,
    dt=deck.output.dt_snapshot,
    threads=cfg.n_threads,
    start=cfg.start_sdf,
    stop=cfg.stop_sdf,
)
for SDF in SDFs:
    SDF.mag *= 1e9  # Convert to nT
    SDF.mid_grid *= 1e-3  # convert to km
cs = hja.CenteredShock(SDFs, deck)

# %%
# Try for just one frame
cs.n_chunks = cfg.n_chunks
_n_lags = 20  # Ideal lags, use n_lags defined below for actual lags
y_idx = cs.grid.y.size // 2

chunk = cs.downstream_start_chunk
t_idx = int(np.argmax(cs.valid_chunks[chunk]))
print(f"Calculating for one frame: ({chunk},{t_idx})")
lags = np.logspace(0, np.log10(cs.chunk_i / 2), _n_lags, dtype=np.int32)
lags = np.unique(lags)
n_lags = len(lags)

print(f"{n_lags} lags: [{' '.join(lags.astype(str))}]")
print(f"Total chunk size: {cs.chunk_i}")


def struc_fn(arr: hj.arrfloat, lags: hj.arrint, order: int) -> hj.arrfloat:
    """Structure function of order `order` for array `arr` at lags `lags`.

    Note:
        arr has shape (n_steps, n_components), i.e. (20, 3) for 20 steps of bx,by,bz.

    Args:
        arr (hj.arrfloat): Array to calculate structure function of.
        lags (hj.arrint): Lags to calculate structure function at.
        order (int): Order of structure function.
    """
    structure = np.empty(lags.shape)
    for i, lag in enumerate(lags):
        dB = np.roll(arr, -lag, axis=0)[:-lag, :] - arr[:-lag, :]
        structure[i] = np.mean(np.sum(dB**order, axis=1))
    return structure


def frame_spectrum(
    cs: hja.CenteredShock,
    chunk: int,
    t_idx: int,
    lags: hj.arrint,
    y_idx: int,
) -> hj.arrfloat:
    """Calculate the spectrum for one frame. Uses a single y-row.

    Args:
        cs (CenteredShock): CenteredShock object.
        chunk (int): Chunk to calculate spectrum for.
        t_idx (int): Time index to calculate spectrum for.
        lags (NDArray[np.int_]): Lags to calculate structure function at.
        y_idx (int): y-index to calculate spectrum for.

    Returns:
        NDArray[np.float64]: Kurtosis for each lag. Same shape as `lags`.
    """

    def get_bn(sdf: hj.sdf_files.SDF, n: str, y_idx: int) -> hj.arrfloat:
        return getattr(sdf.mag, n)[:, y_idx]

    f_bx = partial(get_bn, n="bx", y_idx=y_idx)
    f_by = partial(get_bn, n="by", y_idx=y_idx)
    f_bz = partial(get_bn, n="bz", y_idx=y_idx)

    data = np.empty((cs.chunk_i, 3))
    data[:, 0] = cs.get_qty_in_frame(f_bx, chunk, t_idx)[0]
    data[:, 1] = cs.get_qty_in_frame(f_by, chunk, t_idx)[0]
    data[:, 2] = cs.get_qty_in_frame(f_bz, chunk, t_idx)[0]

    sfn = partial(struc_fn, arr=data, lags=lags)
    spec = sfn(order=4) / (sfn(order=2) ** 2)
    if spec.max() > 1e3 or np.any(np.isinf(spec)):
        spec[:] = np.nan
    return spec


def struc_fn_2d(arr: hj.arrfloat, lags: hj.arrint, order: int) -> hj.arrfloat:
    """2D structure function that averages over all y.

    Note:
        `arr` has shape (x, y, component)

    Args:
        arr (NDArray[np.float64]): Array to calculate structure function of.
        lags (NDArray[np.int_]): Lags to calculate structure function at.
        order (int): Order of structure function.

    Returns:
        NDArray[np.float64]: Structure function at each lag.
    """
    structure = np.empty(lags.shape)
    for i, lag in enumerate(lags):
        dB = np.roll(arr, -lag, axis=0)[:-lag] - arr[:-lag]
        power = dB**order
        averaged_y = power.mean(axis=1)  # (x, y, component) -> (x, component)
        summed_components = averaged_y.sum(axis=1)  # -> (x)
        structure[i] = np.mean(summed_components)  # -> scalar
    return structure


def frame_spectrum_2d(
    cs: hja.CenteredShock,
    chunk: int,
    t_idx: int,
    lags: hj.arrint,
) -> hj.arrfloat:
    """Same as `frame_spectrum` but averages over all y insead of choosing a row.

    Args:
        cs (CenteredShock): CenteredShock object.
        chunk (int): Chunk to calculate spectrum for.
        t_idx (int): Time index to calculate spectrum for.
        lags (NDArray[np.int_]): Lags to calculate structure function at.
        y_idx (int): y-index to calculate spectrum for.

    Returns:
        NDArray[np.float64]: Kurtosis for each lag. Same shape as `lags`.
    """

    def get_bn(sdf: hj.sdf_files.SDF, n: str) -> hj.arrfloat:
        return getattr(sdf.mag, n)

    f_bx = partial(get_bn, n="bx")
    f_by = partial(get_bn, n="by")
    f_bz = partial(get_bn, n="bz")

    data = np.empty((3, cs.chunk_i, cs.grid.y.size))
    data[0] = cs.get_qty_in_frame(f_bx, chunk, t_idx)[0]
    data[1] = cs.get_qty_in_frame(f_by, chunk, t_idx)[0]
    data[2] = cs.get_qty_in_frame(f_bz, chunk, t_idx)[0]
    data = np.moveaxis(data, [0, 1, 2], [2, 0, 1])  # Swap axis to (x, y, component)

    sfn = partial(struc_fn_2d, arr=data, lags=lags)
    spec = sfn(order=4) / (sfn(order=2) ** 2)
    if spec.max() > 1e3 or np.any(np.isinf(spec)):
        spec[:] = np.nan
    return spec


spec = frame_spectrum(cs, chunk, t_idx, lags, y_idx)
ax: Axes
fig, ax = plt.subplots()

ax.plot(lags, spec, color="k", ls=":", alpha=0.8)  # type: ignore
ax.scatter(lags, spec, color="k", marker="x")  # type: ignore
ax.set_xscale("log")

ax.set_xlabel(r"Lag $\ell$ [km]")
ax.set_ylabel(r"Kurtosis $\kappa_S$")

fig.tight_layout()
plt.show()


# %%
# For one time and all chunks
# Times won't be the same for each chunk

spec = np.empty((cs.n_chunks, n_lags))
t_idxs = np.argmax(cs.valid_chunks, axis=1)
chunks = np.arange(cs.n_chunks)
di = deck.constant.di * 1e-3  # km
dists = np.linspace(0, cs.n_chunks, cs.n_chunks + 1) - cs.downstream_start_chunk
dists *= cs.chunk_i * cs.dx / di
lags_wide = hj.arrays.logspaced_edges(lags) * cs.dx / di

for i, (chunk, t_idx) in enumerate(zip(chunks, t_idxs)):
    spec[i, :] = frame_spectrum(cs, chunk, t_idx, lags, y_idx)

fig, ax = plt.subplots()

im = ax.pcolormesh(
    dists,
    lags_wide,
    spec.T,
)
ax.set_yscale("log")

ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel(r"Lag $\ell$ [$d_i$]")
fig.colorbar(im, ax=ax, label=r"Kurtosis $\kappa_S$")

fig.tight_layout()
plt.show()

# %%
# Using mean across y
spec = np.empty((cs.n_chunks, n_lags))
for i, (chunk, t_idx) in enumerate(zip(chunks, t_idxs)):
    spec[i, :] = frame_spectrum_2d(cs, chunk, t_idx, lags)

fig, ax = plt.subplots()
im = ax.pcolormesh(
    dists,
    lags_wide,
    spec.T,
)
ax.set_yscale("log")
ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel(r"Lag $\ell$ [$d_i$]")
fig.colorbar(im, ax=ax, label=r"Kurtosis $\kappa_S$")
fig.tight_layout()
plt.show()

# %%
# Mean across all available times also
chunks = np.arange(cs.n_chunks)
all_t_idxs: list[hj.arrint] = [np.nonzero(cs.valid_chunks[i, :])[0] for i in chunks]

spec = np.empty((cs.n_chunks, n_lags))

start_t = time()
prev_t = start_t
for i in range(cs.n_chunks):
    chunk = chunks[i]
    t_idxs = all_t_idxs[i]

    chunk_spec = np.empty((t_idxs.size, n_lags))
    for j, t_idx in enumerate(t_idxs):
        curr_t = time()
        dt = curr_t - prev_t
        prev_t = curr_t
        print(
            f"Chunk {i:02d}/{cs.n_chunks-1:02d} | Time {j:03d}/{t_idxs.size-1:03d} "
            f"| Elapsed {time()-start_t:06.2f}s [{1/(dt):03.0f}it/s]",
            end="\r",
            flush=True,
        )
        chunk_spec[j, :] = frame_spectrum_2d(cs, chunk, t_idx, lags)
    spec[i, :] = chunk_spec.mean(axis=0)

# %%
axs: list[Axes]
fig, axs = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[96, 4]))  # type: ignore
ax = axs[0]
cax = axs[1]

im = ax.pcolormesh(
    dists,
    lags_wide,
    spec.T,
    norm=LogNorm(vmin=spec.min(), vmax=spec.max()),
)

ax.set_yscale("log")
ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel(r"Lag $\ell$ [$d_i$]")

fig.colorbar(im, cax=cax, label=r"Kurtosis $\kappa_S$")
cax.set_yticks([np.round(spec.min(), 1), 1.5, 2, 2.5, 3, 4])

fig.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()

# %%
# Slopes spectra in chunk


def line(x: hj.arrfloat, m: float, c: float) -> hj.arrfloat:
    return m * x + c


def fit_line(
    spectrogram: hj.arrfloat, lags: hj.arrfloat
) -> tuple[hj.arrfloat, hj.arrfloat]:
    """spectrogram has shape (N, lags).

    Args:
        spectrogram (hj.arrfloat): Spectrogram shape (N_points, lags).
        lags (hj.arrfloat): Lags array.

    Returns:
        slope (hj.arrfloat): Slope of the line fit to each chunk in spectrogram.
        slope_sd (hj.arrfloat): Standard deviation of the slope.
    """
    log_lags = np.log10(lags)
    log_spectrogram = np.log10(spectrogram)
    slope = np.empty(log_spectrogram.shape[0])
    slope_sd = np.empty(log_spectrogram.shape[0])
    for i in range(log_spectrogram.shape[0]):
        popt, pcov = curve_fit(line, log_lags, log_spectrogram[i, :])
        slope[i] = popt[0]
        slope_sd[i] = np.sqrt(np.diag(pcov))[0]
    return slope, slope_sd


def plot_kurt_scaling(
    ax: Axes, dists: hj.arrfloat, slp: hj.arrfloat, slp_sd: hj.arrfloat
) -> None:
    dd = dists[1] - dists[0]
    ax.errorbar(
        dists,
        slp,
        yerr=slp_sd,
        color="k",
        marker=".",
        ls=":",
    )
    ax.set_xlabel("Distance from shock [$d_i$]")
    ax.set_ylabel(r"Slope  $^{\log(\kappa_S)} / _{\log(\ell)}$")
    ax.set_xlim((dists[0] - dd / 2, dists[-1] + dd / 2))


slp, slp_sd = fit_line(spec, lags * cs.dx / di)
dists = (chunks - cs.downstream_start_chunk + 0.5) * cs.chunk_i * cs.dx / di

fig, ax = plt.subplots()
plot_kurt_scaling(ax, dists, slp, slp_sd)
fig.tight_layout()
plt.show()

# %%
# Fit using actual exponential instead of log-transformed


def exp_decay_scale(x, x_ds, A, B):
    return A * np.exp(-x / x_ds) + B


ds_dists = dists[dists > 0]
ds_slp = slp[dists > 0]


def fit_plot_decay(
    ax: Axes, ds_dists: hj.arrfloat, ds_slp: hj.arrfloat
) -> tuple[hj.arrfloat, hj.arrfloat]:
    popt, pcov = curve_fit(
        exp_decay_scale,
        ds_dists,
        ds_slp,
        bounds=([0, 0, -0.1], [50, 1.5, 0.05]),
    )
    perr = np.sqrt(np.diag(pcov))
    print(popt, perr)

    # All combinations of sd's

    all_fits = [[i - j, i, i + j] for i, j in zip(popt, perr)]
    all_fits_prod = list(product(*all_fits))

    fit_x = np.linspace(dists[dists > 0].min() / 2, dists.max(), 50)
    fit = exp_decay_scale(fit_x, *popt)
    fits = np.asarray([exp_decay_scale(fit_x, *f) for f in all_fits_prod])
    fits_min = fits.min(axis=0)
    fits_max = fits.max(axis=0)

    intersect = [popt[0], fit[np.argmin(abs(fit_x - popt[0]))]]

    ax.fill_between(
        fit_x, fits_min, fits_max, edgecolor="none", facecolor=colours.red(), alpha=0.2
    )
    _, cap, _ = ax.errorbar(
        *intersect,
        color=colours.red(),
        markeredgecolor="none",
        markerfacecolor=colours.red(),
        marker=".",
        ls="none",
        xerr=perr[0],
        label=rf"$x_{{ds}}={popt[0]:.2f}\pm{perr[0]:.2f}\ d_i$",
    )
    for c in cap:
        c.set_markeredgecolor(colours.red())
    ax.plot(fit_x, fit, color=colours.red(), ls="--")
    return fit_x, fit


ax: Axes
fig, ax = plt.subplots()
plot_kurt_scaling(ax, dists, slp, slp_sd)
fit_plot_decay(ax, ds_dists, ds_slp)
ax.legend()
fig.tight_layout()
plt.show()

# %%
ds_dists = dists[(dists > 0) & (slp < 0)]
ds_slp = slp[(dists > 0) & (slp < 0)]
fig, ax = plt.subplots()
plot_kurt_scaling(ax, dists, slp, slp_sd)
fit = fit_plot_decay(ax, ds_dists, ds_slp)
ax.axhline(fit[1][-1], ls=":", color=colours.red())
ax.legend()
fig.tight_layout()
plt.show()

# %%


def arrs_csv(arrs: list[npt.NDArray]) -> str:
    import io

    out_arr = np.stack(arrs, axis=1)

    with io.StringIO() as buf:
        np.savetxt(buf, out_arr, delimiter=",", newline="\n", fmt="%.5f")
        out_str = buf.getvalue()

    return out_str


print(arrs_csv([dists, slp]))
# up = slp[1:10].mean()
# dn = fit[1][-1]
# print(abs(up - dn))
# print(up / dn)
