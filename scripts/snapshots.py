# %%
from multiprocessing import Pool, set_start_method
from os import environ
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
from epoch_cheats.deck import Deck
from lic import gen_seed, lic  # type: ignore
from matplotlib.axes import Axes
from matplotlib.colors import SymLogNorm
from phdhelper import mpl
from phdhelper.colours import sim as colours
from tqdm import tqdm  # type: ignore

import hybrid_jp as hj
import hybrid_jp.analysis as hja

# %%
mpl.format()
set_start_method("fork")
cfg = hj.config_from_toml(environ["TOML_PATH"]).shk


# %%
class VisualisationMethod(Protocol):
    def __init__(self, sdf: hj.sdf_files.SDF) -> None:
        ...

    def retrieve_data(self) -> None:
        ...

    def add_data_to_axes(self, ax: Axes, **kwargs) -> None:
        ...


class VisualiseLIC(VisualisationMethod):
    def __init__(self, sdf: hj.sdf_files.SDF) -> None:
        self.sdf = sdf
        self.mag: hj.Mag
        self.grid: hj.Grid

    def retrieve_data(self) -> None:
        self.mag = self.sdf.mag * 1e9  # nT
        self.grid = self.sdf.mid_grid * 1e-3  # km

    def add_data_to_axes(self, ax: Axes, **kwargs) -> None:
        bz = self.mag.bz
        colour_lims = np.array([-np.abs(bz).max(), np.abs(bz).max()]) * 0.75
        ax.pcolormesh(
            self.grid.x,
            self.grid.y,
            bz.T,
            cmap=mpl.Cmaps.diverging,
            vmin=colour_lims[0],
            vmax=colour_lims[1],
        )

        # Generate LIC
        length: int = int(kwargs["length"]) if "length" in kwargs else 100
        seed: np.ndarray | None = np.array(kwargs["seed"]) if "seed" in kwargs else None
        convolution = lic(
            self.mag.bx,
            self.mag.by,
            length=length,
            contrast=False,
            seed=seed,  # type: ignore
        )

        conv = convolution**3

        ax.pcolormesh(
            self.grid.x,
            self.grid.y,
            conv.T,
            cmap=mpl.Cmaps.greyscale_r,
            alpha=0.5,
        )


class VisualiseBField(VisualisationMethod):
    """Visualise B field."""

    def __init__(self, sdf: hj.sdf_files.SDF) -> None:
        self.sdf = sdf

        self.mag: hj.Mag
        self.grid: hj.Grid

    def retrieve_data(self) -> None:
        self.mag = self.sdf.mag * 1e9  # nT
        self.grid = self.sdf.mid_grid * 1e-3  # km

    def add_data_to_axes(self, ax: Axes, **kwargs) -> None:
        # Plot B_z in the bg as the colour
        bz = self.mag.bz
        colour_lims = np.array([-np.abs(bz).max(), np.abs(bz).max()])
        ax.pcolormesh(
            self.grid.x,
            self.grid.y,
            self.mag.bz.T,
            cmap=mpl.Cmaps.diverging,
            vmin=colour_lims[0],
            vmax=colour_lims[1],
        )

        # Plot Bx, By streamlines overtop
        density = kwargs["density"] if "density" in kwargs else 4
        ax.streamplot(
            self.grid.x,
            self.grid.y,
            self.mag.bx.T,
            self.mag.by.T,
            linewidth=0.4,
            color="k",
            arrowstyle="-",
            broken_streamlines=False,
            density=density,
        )


def snapshot(
    sdf_file: Path, deck: Deck, Visualiser: type[VisualisationMethod]
) -> VisualisationMethod:
    sdf = hj.sdf_files.load_sdf_verified(sdf_file, deck.output.dt_snapshot)

    vis = Visualiser(sdf)
    vis.retrieve_data()
    return vis


paths = sorted(list(Path(cfg.data_dir).glob("*.sdf")))
print(cfg.data_dir)
path = paths[len(paths) // 2]
deck = hja.load_deck(cfg.data_dir)
# vis = snapshot(sdf_file=path, deck=deck, Visualiser=VisualiseBField)

nx, ny = int(deck.control.nx), int(deck.control.ny)

axs: list[Axes]
fig, axs = plt.subplots(
    3,
    1,
    sharex=True,
)
# vis.add_data_to_axes(ax)
LICseed = gen_seed((nx, ny))
for i, ax in enumerate(axs):
    snap = [50, 100, 150][i]
    vis = snapshot(sdf_file=paths[snap - 1], deck=deck, Visualiser=VisualiseLIC)
    vis.add_data_to_axes(ax, density=2, length=50, seed=LICseed)
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_title(f"{deck.output.dt_snapshot * (snap - 1):.1f}s")


fig.tight_layout()
plt.show()

# %%
save_dir = Path("/Users/jamesplank/Documents/PHD/hybrid_jp/scripts/snap_frames")
for png in save_dir.glob(f"{cfg.name}-*.png"):
    png.unlink()


def gen_pdf(frame_idx: int):
    frame = paths[frame_idx]
    vis = snapshot(sdf_file=frame, deck=deck, Visualiser=VisualiseLIC)

    fig, ax = plt.subplots()
    vis.add_data_to_axes(ax, seed=LICseed, length=50)
    ax.grid(False)
    ax.annotate(
        f"{frame_idx+1}\n"
        f"{deck.output.dt_snapshot*frame_idx:5.1f} $s$\n"
        rf"{deck.output.dt_snapshot*frame_idx*deck.constant.wci:5.1f} $\Omega_i$",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        bbox=dict(fc="white", ec="none"),
    )
    fig.tight_layout()
    fname = save_dir / f"{cfg.name}-{frame.stem}.png"
    fig.savefig(fname, dpi=300)
    # print(f"Written frame to {fname}")
    fig.clf()
    plt.close()
    del fig, ax


n_paths = len(paths)
with Pool(cfg.n_threads) as pool:
    list(tqdm(pool.imap_unordered(gen_pdf, range(len(paths))), total=n_paths))
