"Proj"
# %%
from multiprocessing import set_start_method
from pathlib import Path
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from phdhelper import mpl
from phdhelper.colours import sim as colours

from hybrid_jp.analysis import CenteredShock, load_deck, load_sdfs_para
from hybrid_jp.sdf_files import SDF

# %%
mpl.format()
set_start_method("fork")

# %%
data_dir = Path().resolve().parent / "U6T40"
START, END = 20, 200
deck = load_deck(data_dir=data_dir)
SDFs, files = load_sdfs_para(
    data_dir,
    threads=7,
    dt=deck.output.dt_snapshot,
    start=START,
    stop=END,
)

cs = CenteredShock(SDFs, deck)

# %%
qty = np.empty((cs.grid_km.x.size, cs.time.size))
for i in range(cs.time.size):
    bx = cs.sdfs[i].mag.bx.mean(axis=1)
    by = cs.sdfs[i].mag.by.mean(axis=1)
    bz = cs.sdfs[i].mag.bz.mean(axis=1)
    btot = np.linalg.norm(np.stack([bx, by, bz], axis=0), axis=0)
    qty[:, i] = btot


def insert_qty_2_shock_arr(cs: CenteredShock, qty: npt.NDArray[np.float64]):
    required_shape = np.asarray([cs.grid_km.x.size, cs.time.size])
    if not np.array_equal(qty.shape, required_shape):
        raise ValueError(
            f"qty has shape {qty.shape} but required shape is {tuple(required_shape)}"
        )
    full_i = cs.dist_either_side.max(axis=0)
    arr = np.empty((cs.full_width, cs.time.size))
    arr[:] = np.nan

    for i in range(cs.time.size):
        insert_i = full_i[0] - cs.dist_either_side[i][0]
        arr[insert_i : insert_i + cs.grid_km.x.size, i] = qty[:, i]

    return arr


# %%
def split_into_chunks(
    cs: CenteredShock,
    n_chunks: int,
) -> tuple[int, npt.NDArray[np.int_]]:
    """Split arr into n_chunks.

    Split arr of shape (N,M) into n_chunks along N axis with a chunk boundary at index
    0 <= align_at_index <= N. The chunks may not cover the whole array, i.e. there may
    be a gap at the start and/or end of arr in order to fit N total chunks.

    Note:
        good options for n_chunks = [10, 23]
    """
    chunk_grow = n_chunks + 1
    chunk_i: int = np.floor(cs.full_width / chunk_grow).astype(int)
    shock_dist = cs.dist_either_side.max(axis=0)
    ratio = shock_dist / shock_dist[1]
    fraction = ratio * (chunk_grow / ratio.sum())
    # rem = np.floor((fraction % 1) * chunk_i).astype(int)
    fraction = np.floor(fraction).astype(int)
    missing: npt.NDArray[np.int_] = shock_dist - fraction * chunk_i
    return chunk_i, missing


def get_valid_chunks(
    cs: CenteredShock,
    n_chunks: int,
    chunk_i: int | None = None,
    missing: npt.NDArray[np.int_] | None = None,
) -> npt.NDArray[np.float64]:
    if chunk_i is None or missing is None:
        chunk_i, missing = split_into_chunks(cs, n_chunks)

    # Array of booleans where data is present
    arr = insert_qty_2_shock_arr(
        cs, np.ones((cs.grid_km.x.size, cs.time.size), dtype=bool)
    )
    arr[np.isnan(arr)] = 0  # Set areas of no data to False instead of NaN
    arr.astype(bool)

    # Bool array same size as number of chunks and number of timesteps
    valid_chunks = np.zeros((n_chunks, cs.time.size), dtype=bool)
    for i in range(cs.time.size):
        valid = np.empty(n_chunks, dtype=bool)
        for j in range(n_chunks):
            cstart = chunk_i * j + missing[0]
            cend = cstart + chunk_i

            if cend > arr.shape[0]:
                raise Exception(
                    f"end index for chunk {j} at timestep {i} is {cend} which is larger"
                    " than the maximum size of {arr.shape[0]}"
                )

            valid[j] = arr[cstart:cend, i].all()
        valid_chunks[:, i] = valid

    return valid_chunks


def plot_chunks(n_chunks: int):
    chunk_i, missing = split_into_chunks(cs, n_chunks)
    valid_chunks = get_valid_chunks(cs, n_chunks, chunk_i, missing)
    eg = insert_qty_2_shock_arr(
        cs, np.stack([sdf.numberdensity.mean(axis=1) for sdf in cs.sdfs], axis=1)
    )
    disp = np.zeros_like(eg, dtype=bool)
    disp[missing[0] : cs.full_width - missing[1], :] = np.repeat(
        valid_chunks, chunk_i, axis=0
    )

    fig, ax = plt.subplots()
    ax.pcolormesh(disp.T)
    ax.pcolormesh(
        eg.T,
        cmap=colours.dark.cmap(),
        alpha=0.3,
    )
    ax.axvline(cs.dist_either_side.max(axis=0)[0], color="k", ls="--")
    ax.set_title(f"{n_chunks} chunks")
    ax.grid(False)
    fig.tight_layout()
    fig.show()


plot_chunks(23)
# %%
# Find chunked data in base arr

n_chunks = 23
chunk_i, missing = split_into_chunks(cs, n_chunks)
valid_chunks = get_valid_chunks(cs, n_chunks, chunk_i, missing)


def get_valid_chunk_data_for_tstamp(
    cs: CenteredShock, arr, chunk_i, missing, valid_chunks
):
    data_start = cs.dist_either_side.max(axis=0)[0] - cs.dist_either_side[:, 0]
    first_chunk = np.argmax(valid_chunks, axis=0) * chunk_i + missing[0]
    start_offset = first_chunk - data_start
    t_chunks = valid_chunks.sum(axis=0)

    out = np.empty_like(arr)
    out[:] = np.nan
    for i in range(cs.time.size):
        so = start_offset[i]
        nc = t_chunks[i]
        if so + nc * chunk_i > arr.shape[0]:
            raise Exception(
                f"end index for chunk {i} at timestep {i} is {so + nc * chunk_i} which"
                "is larger than the maximum size of {arr.shape[0]}"
            )
        out[so : so + nc * chunk_i, i] = arr[so : so + nc * chunk_i, i]
    return out


bx = np.stack([sdf.numberdensity.mean(axis=1) for sdf in cs.sdfs], axis=1)
out = get_valid_chunk_data_for_tstamp(cs, bx, chunk_i, missing, valid_chunks)
plt.pcolormesh(bx.T, alpha=0.6, cmap=colours.dark.cmap())
plt.pcolormesh(out.T, alpha=0.8, cmap=colours.green.cmap())
plt.show()


# %%
def get_valid_chunks_v2(cs: CenteredShock, qty: npt.NDArray[np.float64]):
    cs.n_chunks = 23
    data_start = cs.max_widths[0] - cs.dist_either_side[:, 0]
    first_chunk = np.argmax(cs.valid_chunks, axis=0) * cs.chunk_i + cs.missing[0]
    start_offset = first_chunk - data_start
    total_chunks = cs.valid_chunks.sum(axis=0)

    out = np.empty_like(qty)
    out[:] = np.nan
    for i in range(cs.time.size):
        so = start_offset[i]
        nc = total_chunks[i]
        if so + nc * cs.chunk_i > cs.full_width:
            raise Exception(
                f"End index at timestep {i} is {so + nc * cs.chunk_i} which is larger"
                f" than the maximum size of {cs.full_width}"
            )
        out[so : so + nc * cs.chunk_i, i] = qty[so : so + nc * cs.chunk_i, i]
    return out


nd = np.stack([np.median(sdf.numberdensity, axis=1) for sdf in cs.sdfs], axis=1)
out = get_valid_chunks_v2(cs, nd)
plt.pcolormesh(out.T)
plt.show()


# %%
def get_nd(sdf: SDF) -> npt.NDArray[np.float64]:
    return sdf.numberdensity


def chunk_in_x(cs: CenteredShock, chunk: int, t_idx: int):
    if not cs.valid_chunks[chunk, t_idx]:
        raise ValueError(f"Chunk {chunk} at timestep {t_idx} is not a valid chunk.")
    return chunk * cs.chunk_i + cs.start_offset[t_idx]


def get_qty_in_chunk_at_time(
    cs: CenteredShock,
    qty: Callable[[SDF], npt.NDArray[np.float64]],
    chunk: int,
    t_idx: int,
) -> tuple[npt.NDArray[np.float64], tuple[int, int]]:
    if cs.n_chunks is None:
        raise ValueError("n_chunks must be set before calling this method.")
    if 0 > chunk >= cs.n_chunks:
        raise ValueError(f"{chunk=} must be in the range (0,n_chunks].")

    chunk_offset = chunk_in_x(cs, chunk, t_idx)
    start_stop = (chunk_offset, chunk_offset + cs.chunk_i)
    return qty(cs.sdfs[t_idx])[slice(*start_stop)], start_stop


# data, idx = get_qty_in_chunk_at_time(cs, get_nd, cs.downstream_start_chunk - 1, 64)
# x = (
#     cs.grid_km.x[slice(*idx)]
#     - cs.grid_km.x[cs.downstream_start_chunk * cs.chunk_i + cs.start_offset[64]]
# )
# print(x[0])
# plt.pcolormesh(
#     x,
#     cs.grid_km.y,
#     data.T,
# )
# plt.ylabel("y")
# plt.xlabel("x")
# plt.tight_layout()
# plt.show()

# %%
axs: list[plt.Axes]
fig, axs = plt.subplots(3, 3)  # type: ignore
axs = list(np.asarray(axs).flatten())

ds_chunk = cs.downstream_start_chunk
ax = 0
for i in [50, 51, 52]:
    for j in [ds_chunk - 1, ds_chunk, ds_chunk + 1]:
        data, idx = get_qty_in_chunk_at_time(cs=cs, qty=get_nd, chunk=j, t_idx=i)
        x = cs.grid_km.x[slice(*idx)] - cs.grid_km.x[chunk_in_x(cs, ds_chunk, i)]
        axs[ax].pcolormesh(x, cs.grid_km.y, data.T, vmin=0, vmax=7e7)
        axs[ax].grid(False)
        ax += 1

for ax in range(6):
    axs[ax].set_xticklabels([])
for ax in [1, 2, 4, 5, 7, 8]:
    axs[ax].set_yticklabels([])

fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.show()

# %%
fig, ax = plt.subplots()


def overlay(
    bottom: npt.NDArray[np.float64], top: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    out = np.empty_like(bottom)
    # rescale bottom and top to be between 0 and 1
    bottom = (bottom - bottom.min()) / (bottom.max() - bottom.min())
    top = (top - top.min()) / (top.max() - top.min())

    mask = bottom < 0.5
    out[mask] = (2 * bottom * top)[mask]
    out[~mask] = (1 - 2 * (1 - bottom) * (1 - top))[~mask]

    return out


data = []
n_frames = 2
start_frame = 51
for i in [51, 71]:
    dat, idx = get_qty_in_chunk_at_time(cs, get_nd, ds_chunk, i)
    data.append(dat)
data = np.stack(data, axis=0)
plt.pcolormesh(overlay(data[0], data[1]).T)
plt.show()

# %%
arra = cs.sdfs[170].numberdensity
arrb = cs.sdfs[100].numberdensity

fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 3))  # type: ignore
axs[0].pcolormesh(arra.T)
axs[0].set_aspect("equal")
axs[1].pcolormesh(arrb.T)
axs[1].set_aspect("equal")
axs[2].pcolormesh(overlay(arra, arrb).T)
axs[2].set_aspect("equal")
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
plt.show()
