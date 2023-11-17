# %%
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from phdhelper import mpl
from phdhelper.colours import sim

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
mpl.format()
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
data_path = Path(__file__).parent.parent / "data_cache"
data = np.load(data_path / f"{cfg.name}_corr_lens.npz")

x = data["x"]
chunk_keys = [i for i in data.keys() if "chunk" in i]
chunks: dict[int, hj.arrfloat] = {int(i.split("_")[1]): data[i] for i in chunk_keys}

# chunks = {k: v.mean(axis=0) for k, v in chunks.items()}
corr_lens = np.empty((len(chunks), 3, 2))
for i, c in chunks.items():
    corr_lens[i, :, 0] = c.mean(axis=0)
    corr_lens[i, :, 1] = c.std(axis=0) / np.sqrt(c.shape[0])

ax: Axes
fig, ax = plt.subplots()
labels = [
    r"\parallel",
    r"{\perp i}",
    r"{\perp o}",
]
colours = [sim.red(), sim.green(), sim.blue()]

for i in range(3):
    ax.fill_between(
        x,
        corr_lens[:, i, 0] - corr_lens[:, i, 1],
        corr_lens[:, i, 0] + corr_lens[:, i, 1],
        step="post",
        color=colours[i],
        alpha=0.2,
        edgecolors="none",
        label=rf"$\sigma_{{B {labels[i]}}}$",
    )
    ax.step(
        x,
        corr_lens[:, i, 0],
        label=rf"$B_{{{labels[i]}}}$",
        where="post",
        color=colours[i],
    )

ax.set_xlabel("Distance from shock [$d_i$]")
ax.set_ylabel("Correlation length [$d_i$]")
ax.set_title(cfg.name)
ax.legend(loc="upper left")
fig.tight_layout()
plt.show()
