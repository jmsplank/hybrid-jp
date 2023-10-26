# %%
from functools import partial
from multiprocessing import set_start_method
from os import environ

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
    b_t_xyz = mag.b_xyz_chunk(chunk)
    b_xyz = b_t_xyz.mean(axis=0)

    mean_b = b_xyz.mean(axis=(0, 1))
    unit_b = mean_b / np.linalg.norm(mean_b)

    b_basis = hj.arrays.create_orthonormal_basis_from_vec(unit_b)
    b_rotated = hj.arrays.rotate_arr_to_new_basis(b_basis, b_xyz)

    out = np.empty((b_xyz.shape[0], b_xyz.shape[1], 2), dtype=np.float64)
    out[:, :, 0] = b_rotated[:, :, 0]
    out[:, :, 1] = np.linalg.norm(b_rotated[:, :, 1:], axis=2)

    return out


# %%

mag = MagneticField(cs)
x = np.arange(cs.chunk_i * cs.n_chunks, dtype=np.float64) - (
    cs.chunk_i * cs.downstream_start_chunk
)
x *= cs.dx
dat = np.empty((x.size, 2))
for c in range(cs.n_chunks):
    c_idx = c * cs.chunk_i
    dat[c_idx : c_idx + cs.chunk_i] = b_perp_para_chunk(c, mag).mean(axis=1)

fig, ax = plt.subplots()

ax.plot(x, dat[:, 0], c=colours.green(), label=r"$B_\parallel$")
ax.plot(x, dat[:, 1], c=colours.red(), label=r"$B_\perp$")
ax.legend()
fig.tight_layout()
plt.show()
