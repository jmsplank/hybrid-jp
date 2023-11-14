# %%
import re
from multiprocessing import set_start_method
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy.signal import correlate, correlate2d, correlation_lags  # type: ignore

import hybrid_jp as hj
from hybrid_jp import analysis as hja
from hybrid_jp.analysis.mag import MagneticField


def make_fname_safe(name: str):
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "", s)
    if s in {"", ".", ".."}:
        raise Exception("Could not derive file name from '%s'" % name)
    return s


# %%
mpl.format()
set_start_method("fork")

# %%
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
deck = hja.load_deck(cfg.data_dir)
SDFs, _ = hja.load_sdfs_para(
    cfg.data_dir, deck.output.dt_snapshot, cfg.n_threads, cfg.start_sdf, cfg.stop_sdf
)
DI = deck.constant.di * 1e-3
for SDF in SDFs:
    SDF.mid_grid *= 1e-3 / DI  # d_i
    SDF.mag *= 1e9  # nT
cs = hja.CenteredShock(SDFs, deck)
cs.n_chunks = cfg.n_chunks
field_getter = MagneticField(cs)

# %%
chunk = 18  # cs.downstream_start_chunk
test_b = field_getter.b_xyz_frame(
    chunk,
    int(np.argmax(cs.valid_chunks[chunk])),
)


def plot_strem(ax: Axes, vec: hj.arrfloat, x: hj.arrfloat, y: hj.arrfloat) -> QuadMesh:
    vmin = -np.abs(vec[:, :, 2].max())
    vmax = -vmin
    im = ax.pcolormesh(
        x,
        y,
        vec[:, :, 2].T,
        cmap=mpl.Cmaps.diverging,
        vmin=vmin,
        vmax=vmax,
    )
    ax.streamplot(
        x,
        cs.grid.y,
        vec[:, :, 0].T,
        vec[:, :, 1].T,
        linewidth=0.4,
        color="k",
        arrowstyle="-",
        broken_streamlines=False,
        density=3,
    )
    ax.grid(False)
    return im


ax: Axes
fig, ax = plt.subplots()
x = np.arange(cs.chunk_i) * cs.dx
im = plot_strem(ax, test_b, x, cs.grid.y)
ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel("y [$d_i$]")
fig.colorbar(im, ax=ax, label="$B_z$ [$nT$]")
fig.tight_layout()
plt.show()

# %%
vec = test_b.mean(axis=(0, 1))
vec /= np.linalg.norm(vec)
e2_plane = np.array([0, 0, 1], dtype=np.float64)
basis = hj.arrays.create_orthonormal_basis_from_vec(vec, e2_plane=e2_plane)

rot_b = hj.arrays.rotate_arr_to_new_basis(basis, test_b)

# fig, ax = plt.subplots()
# im = plot_strem(ax, rot_b[:, :, (1, 2, 0)], x, cs.grid.y)
# fig.colorbar(im, ax=ax, label=r"$B_{\parallel}$")
# fig.tight_layout()
# plt.show()


# %%
def corr_2d(arr: hj.arrfloat, dx: float, dy: float):
    arr -= arr.mean()
    lags_x = correlation_lags(arr.shape[0], arr.shape[0], mode="full")
    lags_y = correlation_lags(arr.shape[1], arr.shape[1], mode="full")
    lx, ly = np.meshgrid(lags_x, lags_y)
    correlated = correlate2d(arr, arr, mode="full").T
    correlated = correlated / correlated[(lx == 0) & (ly == 0)]
    correlated = correlated[(lx >= 0) & (ly >= 0)].reshape(arr.shape, order="F")
    lags_x = lags_x[lags_x >= 0]
    lags_y = lags_y[lags_y >= 0]
    lags_x = lags_x * dx
    lags_y = lags_y * dy

    return correlated, lags_x, lags_y


use_c = 0
corr2d, lags_x, lags_y = corr_2d(rot_b[:, :, use_c], cs.dx, cs.dy)

fig, axs = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=[97, 3]))
ax = axs[0]
cax = axs[1]

lags_x = lags_x[1:]
lags_y = lags_y[1:]
corr2d = corr2d[1:, 1:]

vmax = np.abs(corr2d).max()
im = ax.pcolormesh(
    lags_x,
    lags_y,
    corr2d.T,
    cmap=mpl.Cmaps.diverging,
    vmax=vmax,
    vmin=-vmax,
    rasterized=True,
)
cl = ax.contour(lags_x, lags_y, corr2d.T, colors="k", levels=np.arange(10) / 10)
ax.contourf(
    lags_x,
    lags_y,
    corr2d.T,
    levels=[-100, 0, 100],
    hatches=["////", None],
    colors="none",
)
comp = [r"$B_\parallel$", r"$B_{\perp i}$", r"$B_{\perp o}$"]
ax.set_title(f"autocorrelation of {comp[use_c]}")
ax.set_xlabel(r"Lag $\ell_x$ [$d_i$]")
ax.set_ylabel(r"Lag $\ell_y$ [$d_i$]")
ax.grid(False)
ax.set_aspect("equal")
ax.set_yscale("log")
ax.set_xscale("log")

x_0 = 10 ** (np.diff(np.log10(lags_x))[0] / 2) + lags_x[0]
y_0 = 10 ** (np.diff(np.log10(lags_y))[0] / 2) + lags_y[0]
ax.set_xlim(lags_x[0], lags_x[-1])
ax.set_ylim(lags_y[0], lags_y[-1])

fig.colorbar(cl, cax=cax, label="Correlation Length [$d_i$]")
fig.colorbar(im, cax=cax)
fig.tight_layout()
plt.savefig(f"correlation_2d/2D_autocorr_{make_fname_safe(comp[use_c])}.pdf")
plt.show()


# %%
def corr_1d(arr: hj.arrfloat, dx: float) -> tuple[hj.arrfloat, hj.arrfloat]:
    """Correlate 1D signal."""
    arr -= arr.mean()
    lags = correlation_lags(arr.size, arr.size, mode="full")
    correlated = correlate(arr, arr, mode="full")
    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]
    lags = lags * dx

    return lags, correlated


def corr_len_1d(lags: hj.arrfloat, correlated: hj.arrfloat) -> float:
    """Obtain correlation length using integration method."""
    try:
        zero_idx = np.nonzero(correlated <= 0)[0][0]
        corr_upto0 = correlated[:zero_idx]
        lag_upto0 = lags[:zero_idx]
    except IndexError:
        corr_upto0 = correlated
        lag_upto0 = lags

    correlation_length = np.trapz(corr_upto0, lag_upto0)
    return correlation_length


def corr_len_2d(arr: hj.arrfloat, dx: float, dy: float) -> tuple[float, float]:
    cx = corr_len_1d(*corr_1d(arr[:, 0], dx))
    cy = corr_len_1d(*corr_1d(arr[0, :], dy))
    return cx, cy


def make_plot_1d_corr(
    arr: hj.arrfloat,
    delta: float,
    title: str,
    component_name="i",
    save: bool = False,
):
    """_summary_

    Args:
        arr (hj.arrfloat): 1D array to be autocorrelated
        delta (float): spacing between points in `arr`
        title (str): the title of the plot
        component_name (str, optional): subs to B_{component_name}. Defaults to "i".
        save (bool, optional): Save the image. Defaults to False.
    """
    l, c = corr_1d(arr, delta)
    l0 = np.nonzero(c <= 0)[0][0]
    l_C = corr_len_1d(l, c)

    ax: Axes
    fig, ax = plt.subplots()

    ax.plot(l, c)
    ax.scatter(l, c, marker=".", color=colours.dark(), s=10)
    ax.scatter(
        l[l0],
        0,
        marker="x",
        color="red",
        label=f"Zero crossing at ${l[l0]:.2f}d_i$",
    )
    ax.axhline(0, ls="--")
    ax.fill_between(
        l[: l0 + 1],
        0,
        c[: l0 + 1],
        color="red",
        alpha=0.1,
        label=rf"$\lambda_C = {l_C:0.2f}$ [$d_i$]",
    )

    inset = ax.inset_axes((0.5, 0.5, 0.47, 0.47))
    inset.plot(
        np.arange(arr.size)[l0:] * delta,
        (arr - arr.mean())[:-l0],
        color=colours.red(),
        ls="--",
        alpha=0.5,
        label=rf"$B_{component_name}(x\rightarrow x+{l[l0]:.2f}d_i)$",
    )
    inset.plot(np.arange(arr.size) * delta, arr - arr.mean())
    inset.set_xlabel("$x$ [$d_i$]")
    inset.set_ylabel(rf"$B_{component_name} - \left<B_{component_name}\right>$ [$nT$]")
    inset.grid(False)
    inset.legend()

    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.grid(False)
    ax.set_ylabel(r"Autocorrelation")
    ax.set_xlabel(r"Lag $\ell$ [$d_i$]")

    fig.tight_layout()

    if save:
        file_dir = Path(__file__)
        save_dir = file_dir.parent / file_dir.stem
        save_dir.mkdir(exist_ok=True)
        save_name = save_dir / (make_fname_safe(title) + ".png")
        plt.savefig(save_name)

    plt.show()


make_plot_1d_corr(
    test_b[:, 0, 0],
    cs.dx,
    r"1D $B_\parallel$ correlation along $x$",
    component_name=r"\parallel",
    save=True,
)

make_plot_1d_corr(
    test_b[0, :, 0],
    cs.dy,
    r"1D $B_\parallel$ correlation along $y$",
    component_name=r"\parallel",
    save=True,
)

make_plot_1d_corr(
    test_b[:, 0, 1],
    cs.dy,
    r"1D $B_{\perp i}$ correlation along $x$",
    component_name=r"{\perp i}",
    save=True,
)

make_plot_1d_corr(
    test_b[0, :, 1],
    cs.dy,
    r"1D $B_{\perp i}$ correlation along $y$",
    component_name=r"{\perp i}",
    save=True,
)
