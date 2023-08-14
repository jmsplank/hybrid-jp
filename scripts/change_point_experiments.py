"""Change point experiments.

The goal is to discover the best method/dataset or set of transformations
that will most accurately predict transitions from solar wind to shock
transition region to magnetosheath.
"""
# %% Importts
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ruptures as rpt
from cycler import cycler
from epoch_cheats import evaluate_deck, validate_deck
from matplotlib.colors import SymLogNorm
from phdhelper import colours, mpl

from hybrid_jp.sdf_files import load_sdf_verified

mpl.format()

# %% Load data
DATA_PATH = Path().resolve().parent / "U6T40"
fname = DATA_PATH / "0160.sdf"

sdf = load_sdf_verified(fname)
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %%

grid = sdf.mid_grid * (1 / deck.constant.di)
rows = dict(nd=sdf.numberdensity) | sdf.mag.all | sdf.elec.all | sdf.current.all
# del rows["bx"]

axs: list[plt.Axes]
fig, axs = plt.subplots(len(rows), 1, sharex=True, sharey=True, figsize=(7, 10))  # type: ignore

for i, (n, r) in enumerate(rows.items()):
    # axs[i].plot(grid.x, r.mean(axis=1), label=n)
    norm = {}
    if r.min() < 0:
        cmap = mpl.Cmaps.diverging
        rng = max(r.max(), -r.min())
        # norm["vmin"] = -rng
        # norm["vmax"] = rng
        norm["norm"] = SymLogNorm(vmin=-rng, vmax=rng, linthresh=0.005)  # type: ignore
    else:
        cmap = mpl.Cmaps.sequential

    axs[i].pcolormesh(grid.x, grid.y, r.T, rasterized=True, cmap=cmap, **norm)
    axs[i].set_ylabel(n)


plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()


# %%
def normalise(arr: np.ndarray) -> np.ndarray:
    return (arr - arr.min(axis=0)) / (arr.max(axis=0) - arr.min(axis=0))


fig, ax = plt.subplots()
cols = cycler("color", colours.sim.colours_hex + ["red", "green", "blue"])()
for i, (n, r) in enumerate(rows.items()):
    x = normalise(r[:, :10])
    col = next(cols)
    for i in range(x.shape[1]):
        _ = ax.plot(grid.x, x[:, i] - x[0, i], **col, alpha=0.2, ls="-")
        if i == 0:
            _[0].set_label(n)

ax.legend()
plt.tight_layout()
plt.show()

# %%
