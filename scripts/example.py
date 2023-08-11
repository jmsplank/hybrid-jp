"""Example usage of hybrid_jp module."""
# %% Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from epoch_cheats import evaluate_deck, validate_deck
from phdhelper import colours, mpl
from scipy.ndimage import gaussian_filter

from hybrid_jp import Grid
from hybrid_jp.sdf_files import load_sdf_verified

mpl.format()

# %% Load data
DATA_PATH = Path("/Users/jamesplank/Documents/PHD/hybrid_jp/U6T40")
fname = lambda x: (DATA_PATH / f"{x:04}.sdf").resolve()
print(fname(0))

sdf = load_sdf_verified(fname(128))
"""
sdf contains the following attributes:
    - grid: Grid
    - mid_grid: Grid
    - mag: Mag
    - numberdensity: np.ndarray
    - temperature: np.ndarray
"""
deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %% Plot the magnetic field

grid = sdf.grid
grid_m = sdf.mid_grid
grid = Grid(
    x=grid.x / deck.constant.di,
    y=grid.y / deck.constant.di,
)
grid_m = Grid(
    x=grid_m.x / deck.constant.di,
    y=grid_m.y / deck.constant.di,
)
mag = sdf.mag
density = sdf.numberdensity / 100**3

fig, ax = plt.subplots()
ax.grid(False)
density_smoothed = gaussian_filter(density.T, sigma=3)
im = ax.contourf(
    grid_m.x,
    grid_m.y,
    density_smoothed,
    levels=10,
)
plt.colorbar(im, ax=ax, label=r"density [$cm^{-3}$]", pad=0)

ax.set_ylabel("$y/d_i$")
ax.set_xlabel("$x/d_i$")
plt.show()
# %%
