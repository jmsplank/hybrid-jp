# %%
from multiprocessing import set_start_method

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from dotenv import dotenv_values
from phdhelper.colours import sim as colour
from phdhelper.mpl import format
from scipy.signal import correlate, correlation_lags

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
set_start_method("fork")
# %%
# Loading data
env = dotenv_values("../.env")
get_env = lambda s, typ: typ(env[s])
DATA_DIR = get_env("DATA_DIR", str)
SUBDIVISIONS = get_env("SUBDIVISIONS", int)
N_CHUNKS = get_env("N_CHUNKS", int)
N_THREADS = get_env("N_THREADS", int)
START_SDF = get_env("START_SDF", int)
END_SDF = get_env("END_SDF", int)
deck = hja.load_deck(DATA_DIR)
SDFs, fpaths = hja.load_sdfs_para(
    DATA_DIR,
    dt=deck.output.dt_snapshot,
    threads=N_THREADS,
    start=START_SDF,
    stop=END_SDF,
)
for SDF in SDFs:
    SDF.mag *= 1e9  # nT
    SDF.mid_grid *= 1e-3  # km
cs = hja.CenteredShock(SDFs, deck)

# %%
# Average over y (not sure if this is ideal)
bg_B = deck.constant.B0 * 1e9


def get_mean_b(sdf: hj.sdf_files.SDF):
    return np.linalg.norm(np.stack([i for i in sdf.mag], axis=0), axis=0).mean(axis=1)


cs.n_chunks = N_CHUNKS
chnk = cs.downstream_start_chunk
t = int(np.argmax(cs.valid_chunks[chnk]))

mag, _ = cs.get_qty_in_frame(get_mean_b, chnk, t)


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


clen = corr_len_1d(mag, cs.dx)
print(f"Correlation length {clen / (deck.constant.di/1000):.2f}di")
