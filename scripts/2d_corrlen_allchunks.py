# %%
from multiprocessing import Pool
from os import environ
from pathlib import Path

import numpy as np
from phdhelper import mpl
from scipy.ndimage import label  # type: ignore
from scipy.signal import correlate2d, correlation_lags  # type: ignore
from tqdm import trange

import hybrid_jp as hj
from hybrid_jp import analysis as hja
from hybrid_jp.analysis.mag import MagneticField

# %%
mpl.format()
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
deck = hja.load_deck(cfg.data_dir)
SDFs, _ = hja.load_sdfs_para(
    cfg.data_dir,
    deck.output.dt_snapshot,
    cfg.n_threads,
    cfg.start_sdf,
    cfg.stop_sdf,
    ipython=True,
)
DI = deck.constant.di * 1e-3
for SDF in SDFs:
    SDF.mid_grid *= 1e-3 / DI  # d_i
    SDF.mag *= 1e9  # nT
cs = hja.CenteredShock(SDFs, deck)
cs.n_chunks = cfg.n_chunks
field_getter = MagneticField(cs)

# %%


def rotate(arr: hj.arrfloat):
    vec = arr.mean(axis=(0, 1))
    vec /= np.linalg.norm(vec)
    e2_plane = np.array([0, 0, 1], dtype=np.float64)
    basis = hj.arrays.create_orthonormal_basis_from_vec(vec, e2_plane=e2_plane)
    rot_b = hj.arrays.rotate_arr_to_new_basis(basis, arr)
    return rot_b


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


def corr_len(arr: hj.arrfloat, dx: float, dy: float):
    above_zero = arr >= 0
    labelled_area, _ = label(above_zero)  # type: ignore
    corr_area = labelled_area == labelled_area[0, 0]
    floor_area = corr_area * dx * dy
    corr_area = floor_area * arr
    corr_volume = corr_area.sum()
    corr_len = np.sqrt(corr_volume)
    return corr_len


def calc_corr_len(arr: hj.arrfloat, dx: float, dy: float) -> hj.arrfloat:
    rotated = rotate(arr)
    correlation_lengths = np.empty(3)
    for i in range(correlation_lengths.size):
        corr, lx, ly = corr_2d(rotated[:, :, i], dx, dy)
        dx = lx[1] - lx[0]
        dy = ly[1] - ly[0]
        correlation_lengths[i] = corr_len(corr, dx, dy)
    return correlation_lengths


# %%
dx, dy = cs.dx, cs.dy


def process_chunk(chunk_idx: int):
    print(f"{chunk_idx}\n", end=None)
    chunk_data = field_getter.b_xyz_chunk(chunk_idx)
    n_times = chunk_data.shape[0]
    out = np.empty((n_times, 3))
    for tstep in range(n_times):
        out[tstep, :] = calc_corr_len(chunk_data[tstep, :, :], dx, dy)
    print(f"        {chunk_idx}\n", end=None)
    return out


with Pool(7) as pool:
    processed_chunks = list(pool.map(process_chunk, range(cs.n_chunks)))

# %%
saved_data_path = Path(__file__).parent.parent / "data_cache"
data_as_dict = {
    f"chunk_{i}": arr for i, arr in zip(range(cs.n_chunks), processed_chunks)
}
x = (
    (np.arange(cs.n_chunks, dtype=np.float64) - cs.downstream_start_chunk)
    * cs.chunk_i
    * cs.dx
)
data_as_dict["x"] = x
np.savez_compressed(saved_data_path / f"{cfg.name}_corr_lens.npz", **data_as_dict)

# %%
