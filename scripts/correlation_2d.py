# %%
from multiprocessing import set_start_method
from os import environ

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from phdhelper import mpl

# from phdhelper.colours import sim as colurs
from scipy.signal import correlate2d, correlation_lags  # type: ignore

import hybrid_jp as hj
from hybrid_jp import analysis as hja
from hybrid_jp.analysis.mag import MagneticField

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
plt.savefig(f"correlation_2d/2D_autocorr_{hj.make_fname_safe(comp[use_c])}.pdf")
plt.show()
