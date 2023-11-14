# %%
from os import environ

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from phdhelper import mpl

import hybrid_jp as hj

# %%
mpl.format()
snap_sdf_idx = 100
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
sdf_path = sorted(list(cfg.data_dir.glob("*.sdf")))[snap_sdf_idx]
sdf = hj.sdf_files.load_sdf_verified(sdf_path)

data = np.stack([m for m in sdf.mag], axis=2, dtype=np.float64)
ndens = sdf.numberdensity
data[:, :, 2] = ndens  # replace Bz with numberdensity
x_subset = slice(250, 700)
sub_dat = data[x_subset, :, :]


def plot(ax: Axes, data: hj.arrfloat):
    # vmax = np.abs(data).max()
    ax.pcolormesh(
        data[:, :, 2].T,
        # vmin=-vmax,
        # vmax=vmax,
        # cmap=mpl.Cmaps.diverging,
        rasterized=True,
    )
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    ax.streamplot(
        x,
        y,
        data[:, :, 0].T,
        data[:, :, 1].T,
        color="k",
        integration_direction="both",
        broken_streamlines=False,
        linewidth=0.5,
        arrowstyle="-",
        density=4,
    )


ax: Axes
fig, ax = plt.subplots()
plot(ax, sub_dat)
ax.grid(False)
fig.tight_layout()
plt.savefig("snapshot.pdf")
plt.show()
