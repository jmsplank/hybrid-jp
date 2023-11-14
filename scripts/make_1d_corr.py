# %%
from multiprocessing import set_start_method
from os import environ
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from phdhelper import mpl
from phdhelper.colours import sim as colours
from scipy.signal import correlate, correlation_lags  # type: ignore

import hybrid_jp as hj
import hybrid_jp.analysis as hja
from hybrid_jp.analysis.mag import MagneticField

set_start_method("fork", force=True)
mpl.format()
# %%


def corr_1d(arr: hj.arrfloat, dx: float) -> tuple[hj.arrfloat, hj.arrfloat]:
    """Correlate 1D signal."""
    arr -= arr.mean()
    lags = correlation_lags(arr.size, arr.size, mode="full")
    correlated = correlate(arr, arr, mode="full")
    correlated = correlated / correlated[lags == 0]
    correlated = correlated[lags >= 0]
    lags = lags[lags >= 0]
    lags = lags * dx

    return lags, correlated


def corr_len_1d(lags: hj.arrfloat, correlated: hj.arrfloat) -> float:
    """Obtain correlation length using integration method."""
    try:
        zero_idx = np.nonzero(correlated <= 0)[0][0]
        corr_upto0 = correlated[:zero_idx]
        lag_upto0 = lags[:zero_idx]
    except IndexError:
        corr_upto0 = correlated
        lag_upto0 = lags

    correlation_length = np.trapz(corr_upto0, lag_upto0)
    return correlation_length


def make_plot_1d_corr(
    arr: hj.arrfloat,
    delta: float,
    title: str,
    component_name="i",
    save: bool = False,
):
    """_summary_

    Args:
        arr (hj.arrfloat): 1D array to be autocorrelated
        delta (float): spacing between points in `arr`
        title (str): the title of the plot
        component_name (str, optional): subs to B_{component_name}. Defaults to "i".
        save (bool, optional): Save the image. Defaults to False.
    """
    l, c = corr_1d(arr, delta)
    l0 = np.nonzero(c <= 0)[0][0]
    l_C = corr_len_1d(l, c)

    ax: Axes
    fig, ax = plt.subplots()

    ax.plot(l, c)
    ax.scatter(l, c, marker=".", color=colours.dark(), s=10)
    ax.scatter(
        l[l0],
        0,
        marker="x",
        color="red",
        label=f"Zero crossing at ${l[l0]:.2f}d_i$",
    )
    ax.axhline(0, ls="--")
    ax.fill_between(
        l[: l0 + 1],
        0,
        c[: l0 + 1],
        color="red",
        alpha=0.1,
        label=rf"$\lambda_C = {l_C:0.2f}$ [$d_i$]",
    )

    inset = ax.inset_axes((0.5, 0.5, 0.47, 0.47))
    inset.plot(
        np.arange(arr.size)[l0:] * delta,
        (arr - arr.mean())[:-l0],
        color=colours.red(),
        ls="--",
        alpha=0.5,
        label=rf"$B_{component_name}(x\rightarrow x+{l[l0]:.2f}d_i)$",
    )
    inset.plot(np.arange(arr.size) * delta, arr - arr.mean())
    inset.set_xlabel("$x$ [$d_i$]")
    inset.set_ylabel(rf"$B_{component_name} - \left<B_{component_name}\right>$ [$nT$]")
    inset.grid(False)
    inset.legend()

    ax.legend(loc="upper left")
    ax.set_title(title)
    ax.grid(False)
    ax.set_ylabel(r"Autocorrelation")
    ax.set_xlabel(r"Lag $\ell$ [$d_i$]")

    fig.tight_layout()

    if save:
        file_dir = Path(__file__)
        save_dir = file_dir.parent / file_dir.stem
        save_dir.mkdir(exist_ok=True)
        save_name = save_dir / (hj.make_fname_safe(title) + ".png")
        plt.savefig(save_name)

    plt.show()


cfg = hj.config_from_toml(environ["TOML_PATH"]).shk
deck = hja.load_deck(cfg.data_dir)
SDFs, fnames = hja.load_sdfs_para(
    cfg.data_dir,
    deck.output.dt_snapshot,
    cfg.n_threads,
    cfg.start_sdf,
    cfg.stop_sdf,
)
for sdf in SDFs:
    sdf.mag *= 1e9
    sdf.mid_grid *= 1 / deck.constant.di
cs = hja.CenteredShock(SDFs, deck)
cs.n_chunks = cfg.n_chunks

test_chunk = 18
time_idx = int(np.argmax(cs.valid_chunks[test_chunk]))
field_getter = MagneticField(cs)
test_b = field_getter.b_xyz_frame(test_chunk, time_idx)


make_plot_1d_corr(
    test_b[:, 0, 0],
    cs.dx,
    r"1D $B_\parallel$ correlation along $x$",
    component_name=r"\parallel",
    save=True,
)

make_plot_1d_corr(
    test_b[0, :, 0],
    cs.dy,
    r"1D $B_\parallel$ correlation along $y$",
    component_name=r"\parallel",
    save=True,
)

make_plot_1d_corr(
    test_b[:, 0, 1],
    cs.dy,
    r"1D $B_{\perp i}$ correlation along $x$",
    component_name=r"{\perp i}",
    save=True,
)

make_plot_1d_corr(
    test_b[0, :, 1],
    cs.dy,
    r"1D $B_{\perp i}$ correlation along $y$",
    component_name=r"{\perp i}",
    save=True,
)
