# %%
from multiprocessing import set_start_method
from os import environ

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from phdhelper import mpl
from phdhelper.colours import sim as colours

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
set_start_method("fork")
mpl.format()

# %%
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk

deck = hja.load_deck(data_dir=cfg.data_dir)
SDFs, fpaths = hja.load_sdfs_para(
    sdf_dir=cfg.data_dir,
    dt=deck.output.dt_snapshot,
    threads=cfg.n_threads,
    start=cfg.start_sdf,
    stop=cfg.stop_sdf,
)

# %%
for SDF in SDFs:
    SDF.mag *= 1e9  # nT
    SDF.mid_grid *= 1e-3  # km
    SDF.numberdensity *= 1e-6
cs = hja.CenteredShock(sdfs=SDFs, deck=deck)

# %%
# Plot first timestep
axs: list[Axes]
fig, axs = plt.subplots(2, 2)
axs = np.asarray(axs).flatten()  # type: ignore

frames = np.linspace(0, len(SDFs) - 1, 4, dtype=int)
for ax, f in zip(axs, frames):
    sdf = SDFs[f]
    ax.set_title(rf"$t={cs.time[f]:.2f}\ s$")
    ax.pcolormesh(sdf.mid_grid.x, sdf.mid_grid.y, sdf.numberdensity.T)


fig.tight_layout()
plt.show()

# %%
# Visualise the chunks

cs.n_chunks = cfg.n_chunks
valids = np.nonzero(cs.valid_chunks)
l_valids = valids[0].size


def get_med_nd(sdf: hj.sdf_files.SDF):
    return np.median(sdf.numberdensity, axis=1)


x_size = cs.chunk_i
nds_aligned = np.empty((x_size * cs.n_chunks, cs.time.size))
nds_aligned[:] = np.nan
for x, t in zip(*valids):
    nd, _ = cs.get_qty_in_frame(get_med_nd, x, t)
    nds_aligned[x * x_size : x * x_size + x_size, t] = nd

ax: Axes
fig, ax = plt.subplots()
ax.pcolormesh(nds_aligned.T)

fig.tight_layout()
plt.show()
