"""Example usage of hybrid_jp module."""
# %% Imports
from pathlib import Path
from typing import Literal, NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from epoch_cheats import evaluate_deck, validate_deck
from phdhelper import colours, mpl
from scipy.ndimage import gaussian_filter

from hybrid_jp import Grid, Mag
from hybrid_jp.change_points import binseg
from hybrid_jp.sdf_files import load_sdf_verified

mpl.format()

# %% Load data
DATA_PATH = Path("/Users/jamesplank/Documents/PHD/hybrid_jp/U6T40")
fname = lambda x: (DATA_PATH / f"{x:04}.sdf").resolve()
print(fname(0))

sdf = load_sdf_verified(fname(128))
# sdf contains the following attributes:
#     - grid: Grid
#     - mid_grid: Grid
#     - mag: Mag
#     - numberdensity: np.ndarray
#     - temperature: np.ndarray

deck = validate_deck(evaluate_deck(DATA_PATH / "input.deck"))

# %% Plot the numberdensity

grid_m = sdf.mid_grid
grid_m *= 1 / deck.constant.di
# Change density to cm^-3
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


# %% Get the change points

segmentation = binseg(density, nseg=3)
print("Index: ", ", ".join([f"{x:03d}" for x in segmentation]))
# Index:  645, 795, 835
# Use first as start of STR and last as end of STR to get the biggest
# transition region.
force = 1285

# %% Find a custom STR

fig, ax = plt.subplots()
# ax.plot(sdf.mid_grid.x, np.linalg.norm(sdf.mag.bx, axis=1))
ax.plot(np.arange(sdf.mid_grid.x.size), np.linalg.norm(sdf.mag.bx, axis=1))
for i in segmentation:
    ax.axvline(i, color=colours.sim.red(), label=f"{i}")

ax.xaxis.set_major_locator(mticker.MultipleLocator(50))
ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
# ax.grid(True, which="both", axis="x")
for t in ax.get_xticklabels():
    t.set_rotation(90)

ax.axvline(force, color=colours.sim.blue(), label=f"Forced {force}")

ax.legend()
plt.show()

# %%
segmentation = [segmentation[0], force]


# Define some data types
class Data(NamedTuple):
    """Data for a region."""

    grid: Grid
    mag: Mag

    def gen_psd_x(self, y_method: Literal["mean", "slice"] = "mean"):
        """Generate the PSD for each component of the magnetic field.

        Args:
            y_method ("mean"|"slice): Method for dealing with y dimension. Defaults to "mean".

        Raises:
            NotImplementedError: y_method="slice" not implemented yet

        Returns:
            np.ndarray: Frequency array.
            np.ndarray: Power spectral density.
        """
        # TODO: Implement slice method
        if y_method == "slice":
            raise NotImplementedError("y_mehod='slice' not implemented yet")

        dx = self.grid.x[1] - self.grid.x[0]
        freq = np.fft.fftfreq(self.grid.x.size, d=dx)

        # Iterate over bx, by, bz
        # Claculate the power in each component
        _Y = []
        for i in [getattr(self.mag, x) for x in ["bx", "by", "bz"]]:
            y = i.mean(axis=1)
            y = (y - y.mean()) * np.hanning(y.shape[0])
            y = np.fft.fft(y, axis=0)  # y in T
            _Y.append((np.abs(y * 1e9) ** 2) / dx)  # Power in nT^2 m^-1

        # Sum the power in each component
        Y = np.asarray(_Y)
        Y = np.sum(Y, axis=0)
        # Retuern only the positive frequencies
        return freq[freq > 0], Y[freq > 0]


class Regions(NamedTuple):
    """Collection of regions."""

    sw: Data
    str: Data
    ms: Data


regions = Regions(
    sw=Data(
        grid=grid_m.slice_x(0, segmentation[0]),
        mag=sdf.mag.slice_x(0, segmentation[0]),
    ),
    str=Data(
        grid=grid_m.slice_x(segmentation[0], segmentation[-1]),
        mag=sdf.mag.slice_x(segmentation[0], segmentation[-1]),
    ),
    ms=Data(
        grid=grid_m.slice_x(segmentation[-1], grid_m.shape[0]),
        mag=sdf.mag.slice_x(segmentation[-1], grid_m.shape[0]),
    ),
)

# %% Test the regions

fig, ax = plt.subplots()

for i, (region, label) in enumerate(zip(regions, ["SW", "STR", "MS"])):
    ax.plot(region.grid.x, np.linalg.norm(region.mag.bx, axis=1), label=label)

ax.legend()
ax.set_ylabel(r"$B_x$ [T]")
ax.set_xlabel("$x [d_i]$")
plt.show()

# %% Calculate the PSD for each region

psds = [i.gen_psd_x() for i in regions]


fig, ax = plt.subplots()

for i, (freq, psd) in enumerate(psds):
    ax.loglog(freq, psd, label=["SW", "STR", "MS"][i])

# Plot di
ax.axvline(
    1,
    color=colours.sim.blue(),
    linestyle="--",
    label=r"$d_i^{-1}$",
)

ax.set_ylabel(r"PSD [$nT^2 d_i^{-1}$]")
ax.set_xlabel("wavenumber [$d_i^{-1}$]")
ax.legend()

plt.show()
