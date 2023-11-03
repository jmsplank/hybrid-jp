# %%
import json
from functools import partial
from multiprocessing import set_start_method
from os import environ
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy.signal import correlate, correlation_lags  # type: ignore

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
set_start_method("fork")
mpl.format()

# %%
# Loading data
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk

deck = hja.load_deck(cfg.data_dir)
SDFs, fpaths = hja.load_sdfs_para(
    cfg.data_dir,
    dt=deck.output.dt_snapshot,
    threads=cfg.n_threads,
    start=cfg.start_sdf,
    stop=cfg.stop_sdf,
)
for SDF in SDFs:
    SDF.mag *= 1e9  # nT
    SDF.mid_grid *= 1e-3  # km
cs = hja.CenteredShock(SDFs, deck)
cs.n_chunks = cfg.n_chunks


# %%
def corr_len_1d(arr: npt.NDArray, dx: float):
    arr -= arr.mean
    lags = correlation_lags(arr.size, arr.size, mode="full")
    correlated = correlate(arr, arr, mode="full")
    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]
    lags = lags * dx

    try:
        zero_idx = np.nonzero(correlated <= 0)[0][0]
        corr_upto0 = correlated[:zero_idx]
        lag_upto0 = lags[:zero_idx]
    except IndexError:
        corr_upto0 = correlated
        lag_upto0 = lags

    correlation_length = np.trapz(corr_upto0, lag_upto0)
    return correlation_length


class MagneticField:
    def __init__(self, cs: hja.CenteredShock) -> None:
        self.cs = cs
        self.chunk_i = self.cs.chunk_i
        self.ny = self.cs.grid.y.size

    @staticmethod
    def get_mag_xyz(sdf: hj.sdf_files.SDF, q: str) -> hj.arrfloat:
        return getattr(sdf.mag, q)

    f_x = partial(get_mag_xyz, q="bx")
    f_y = partial(get_mag_xyz, q="by")
    f_z = partial(get_mag_xyz, q="bz")
    funcs = [f_x, f_y, f_z]

    def b_xyz_frame(
        self,
        chunk: int,
        t_idx: int,
    ) -> hj.arrfloat:
        out = np.empty((self.chunk_i, self.ny, 3), dtype=np.float64)
        for i, f in enumerate(self.funcs):
            out[:, :, i], _ = self.cs.get_qty_in_frame(
                qty_func=f, chunk=chunk, t_idx=t_idx
            )
        return out

    def b_xyz_chunk(self, chunk: int) -> hj.arrfloat:
        times = np.nonzero(self.cs.valid_chunks[chunk])[0]
        out = np.empty((times.size, self.chunk_i, self.ny, 3), dtype=np.float64)
        for i, time in enumerate(times):
            out[i] = self.b_xyz_frame(chunk=chunk, t_idx=time)

        return out


def b_perp_para_chunk(chunk: int, mag: MagneticField) -> hj.arrfloat:
    b_xyz = mag.b_xyz_chunk(chunk).reshape((mag.ny, -1, 3))
    mean_b = b_xyz.mean(axis=(0, 1))
    unit_b = mean_b / np.linalg.norm(mean_b)
    b_basis = hj.arrays.create_orthonormal_basis_from_vec(
        unit_b, e2_plane=np.array([0, 0, 1], dtype=np.float64)
    )
    b_rotated = hj.arrays.rotate_arr_to_new_basis(b_basis, b_xyz)
    return b_rotated


# %%

mag = MagneticField(cs)
x = np.arange(cs.chunk_i * cs.n_chunks, dtype=np.float64) - (
    cs.chunk_i * cs.downstream_start_chunk
)
x *= cs.dx
chunks: list[hj.arrfloat] = []
for i in range(cs.n_chunks):
    ny = cs.valid_chunks[i].sum() * cs.grid.y.size
    # c_idx = c * cs.chunk_i
    chunks.append(b_perp_para_chunk(i, mag))

# %%
corr = corr_len_1d(chunks[cs.downstream_start_chunk][:, 0, 0], cs.dx)

# %%
out: list[hj.arrfloat] = []
for i, c in enumerate(chunks):  # for each chunk
    corrs = np.empty((c.shape[1], 3))
    for step in range(c.shape[1]):  # Iterate over grid.y
        for component in range(3):  # Iterate over b_para, b_perp1, b_perp2
            corrs[step, component] = corr_len_1d(c[:, step, component], cs.dx)
    out.append(corrs)

# %%
# print(*[i.shape for i in out], sep="\n")
ax: Axes
fig, ax = plt.subplots()

d = np.empty((cs.n_chunks, 3))
for i, corr in enumerate(out):
    d[i, :] = corr.mean(axis=0) / (deck.constant.di * 1e-3)

plot_x: hj.arrfloat = (
    (np.arange(cs.n_chunks) - cs.downstream_start_chunk)
    * cs.dx
    * cs.chunk_i
    / (deck.constant.di * 1e-3)
)
labels = [
    r"$\kappa_\parallel$",
    r"$\kappa_{\perp i}$ in-plane",
    r"$\kappa_{\perp o}$ out-of-plane",
]
ax.step(plot_x, d[:, 0], where="post", label=labels[0])
ax.step(plot_x, d[:, 1], where="post", label=labels[1])
ax.step(plot_x, d[:, 2], where="post", label=labels[2])
ax.legend()
ax.set_title(f"{cfg.name} - correlation length")
fig.tight_layout()
plt.show(block=False)

# %%
data = dict(
    x=plot_x.tolist(),
    y=d.tolist(),
    labels=labels,
    xlabel=r"Distance from shock [$d_i$]",
    ylabel=r"Correlation length $\kappa$ [$d_i$]",
)

content_dir = Path(__file__).parent / "correlation_len"
content_dir.mkdir(parents=False, exist_ok=True)


with open(content_dir / f"{cfg.name}-plot_data.json", "w") as file:
    json.dump(data, file)


# %%
# Combi-plot
def load_data(fp: Path | str) -> dict[str, Any]:
    if not isinstance(fp, Path):
        fp = Path(fp)

    with open(fp, "r") as file:
        data = json.load(file)

    data["x"] = np.array(data["x"])
    data["y"] = np.array(data["y"])

    return data


def plot_data(ax: Axes, data: dict[str, Any]):
    x = data["x"]
    y = data["y"]
    labels = data["labels"]
    for i in range(y.shape[1]):
        ax.step(x, y[:, i], where="post", label=labels[i])


saved_data = list(content_dir.glob("*-plot_data.json"))
data_ref = load_data(saved_data[0])
n_shocks = len(saved_data)

axs: list[Axes]
fig, axs = plt.subplots(1, n_shocks, sharex=True)
if n_shocks == 1:
    axs = [axs]  # type: ignore
for i, (ax, fp) in enumerate(zip(axs, saved_data)):
    data = load_data(fp)
    plot_data(ax, data)
    ax.set_title(fp.stem.split("-")[0])

axs[2].legend()
axs[0].set_ylabel(data_ref["ylabel"])
axs[1].set_xlabel(data_ref["xlabel"])

axs[1].set_yticklabels([])
axs[2].set_yticklabels([])


fig.tight_layout()
fig.subplots_adjust(wspace=0)
plt.show()
