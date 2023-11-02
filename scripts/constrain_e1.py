# %%
from os import environ
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from epoch_cheats.deck import Deck
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
from phdhelper import mpl
from phdhelper.colours import sim

import hybrid_jp as hj
import hybrid_jp.analysis as hja


# %%
def prelim() -> tuple[hj.config.ShockParams, list[Path]]:
    cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
    sdfs = sorted(list(Path(cfg.data_dir).glob("*.sdf")))
    return cfg, sdfs


def load(sdf_path: Path, cfg: hj.config.ShockParams) -> tuple[Deck, hj.arrfloat]:
    # mpl.format()
    deck = hja.load_deck(cfg.data_dir)
    sdf = hj.sdf_files.load_sdf_verified(sdf_path)
    o_mag: hj.Mag = sdf.mag * 1e9
    mag: np.ndarray = np.stack([*o_mag], axis=2)
    return deck, mag


# %%
def calc(mag: hj.arrfloat) -> hj.arrfloat:
    v = mag.mean(axis=(0, 1))
    v = v / np.linalg.norm(v)
    print(v)
    plane_norm = np.array([0, 0, 1])
    e2 = np.cross(v, plane_norm)
    e2 /= np.linalg.norm(e2)

    basis = hj.arrays.create_orthonormal_basis_from_vec(v, e2)
    return basis


# %%
def make_plot():
    fig = plt.figure(figsize=(5, 5))
    ax: Axes3D = fig.add_subplot(projection="3d")  # type: ignore

    ax.set_aspect("equal")
    ax.set_xlim((-1.5, 1.5))
    ax.set_ylim((-1.5, 1.5))
    ax.set_zlim((-1.5, 1.5))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    return fig, ax


def plot(ax: Axes3D, basis: hj.arrfloat) -> None:
    def plot_vector(ax: Axes3D, vec: hj.arrfloat, **q_kwargs):
        ax.quiver(0, 0, 0, *vec, **q_kwargs)

    plot_vector(ax, basis[0], color=sim.red(), label="$e_1$")
    plot_vector(ax, basis[1], color=sim.green(), label="$e_2$")
    plot_vector(ax, basis[2], color=sim.blue(), label="$e_3$")
    plt.pause(3)
    plt.show(block=False)


def eternal(ax: Axes3D, sdfs: list[Path], cfg: hj.config.ShockParams):
    frame_idx = int(input(f"Please choose an sdf 0-{len(sdfs)-1}:"))
    _, mag = load(sdfs[frame_idx], cfg)
    basis = calc(mag)
    plot(ax, basis)


def main():
    matplotlib.use("TkAgg")
    print(f"Backend: {matplotlib.get_backend()}")
    cfg, sdfs = prelim()
    print(f"Loading from {cfg.data_dir}")
    print(f"There are {len(sdfs)} sdf files.")
    frame_idx = int(input(f"Please choose an sdf 0-{len(sdfs)-1}:"))
    dec, mag = load(sdfs[frame_idx], cfg)
    basis = calc(mag)
    fig, ax = make_plot()
    plot(ax, basis)
    ax.legend()

    while True:
        eternal(ax, sdfs, cfg)


def arbitrary():
    vec_str = input("gimme a 3 vector, 3 floats separated by spaces\nhere: ")
    vec_list = [i for i in vec_str.strip().split(" ")]
    vec = np.array(vec_list, dtype=np.float64)
    vec /= np.linalg.norm(vec)
    print(f"Unit vector {vec}")

    e2 = np.cross(vec, np.array([0, 0, 1]))
    e2 /= np.linalg.norm(e2)

    basis = hj.arrays.create_orthonormal_basis_from_vec(vec, e2)
    print(f"basis {np.linalg.norm(basis, axis=0)}")
    return basis


def main2():
    matplotlib.use("TkAgg")
    mpl.format()
    fig, ax = make_plot()
    legend_present = False
    while True:
        basis = arbitrary()
        plot(ax, basis)
        if not legend_present:
            ax.legend()
            legend_present = True


if __name__ == "__main__":
    main2()
