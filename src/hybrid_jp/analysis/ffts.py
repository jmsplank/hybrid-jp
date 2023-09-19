from functools import partial
from typing import Literal

import numpy as np
import numpy.typing as npt

from ..dtypes import arrfloat
from ..sdf_files import SDF
from .shock_centering import CenteredShock


def hann_2d(nx: int, ny: int) -> npt.NDArray[np.float64]:
    """Create a 2d hanning window.

    https://stackoverflow.com/a/65948798

    Example:
        >>> h2d = hann_2d(66, 160)
        >>> X, Y = np.meshgrid(np.arange(66), np.arange(160))
        >>> fig = plt.figure()
        >>> ax = plt.axes(projection="3d")
        >>> ax.contour3D(X, Y, h2d.T, 50)
        >>> plt.show()
    """
    hann_1d = [np.hanning(i) for i in (nx, ny)]
    return np.sqrt(np.outer(*hann_1d))


def power_xy(
    arr: arrfloat,
    dx: float,
    dy: float,
) -> tuple[arrfloat, arrfloat, arrfloat]:
    """Get power spectrum of an array f(x, y).

    Args:
        arr (npt.NDArray[np.float64]): Array of shape (nx, ny)
        dx (float): Spacing between x points
        dy (float): Spacing between y points

    Returns:
        Pxy (npt.NDArray[np.float64]): Power spectrum of arr
        fx (npt.NDArray[np.float64]): x frequencies
        fy (npt.NDArray[np.float64]): y frequencies
    """
    fx = np.fft.fftfreq(arr.shape[0], dx)
    fy = np.fft.fftfreq(arr.shape[1], dy)

    # Center the data and apply Hanning window
    arr = (arr - arr.mean()) * hann_2d(*arr.shape)
    kxky = np.fft.fft2(arr)
    Pxy = np.abs(kxky) ** 2 / (dx * dy)

    # Mask out the negative frequencies
    # expand_dims is needed to broadcast the masks over the correct dimensions in 2d
    # The x mask needs to cover all y so add an extra dimension at axis 1 (y)
    fx_mask = fx > 0
    fx_mask2 = np.expand_dims(fx_mask, axis=1)  # shape (nx, 1)
    # The y mask needs to cover all x so add an extra dimension at axis 0 (x)
    fy_mask = fy > 0
    fy_mask2 = np.expand_dims(fy_mask, axis=0)  # shape (1, ny)

    fx = fx[fx_mask]
    fy = fy[fy_mask]
    Pxy = Pxy[fx_mask2 & fy_mask2].reshape(fx.size, fy.size)

    return Pxy, fx, fy


def subdivide_repeat(
    subdivisons: int,
    arrxy: arrfloat,
    x: arrfloat,
    y: arrfloat,
) -> tuple[arrfloat, arrfloat, arrfloat]:
    """Subdivide an array and repeat the values.

    Args:
        subdivisons (int): Number of subdivisions in each dimension.
        arrxy (NDArray[np.float64]): Array to subdivide.
        x (NDArray[np.float64]): x values.
        y (NDArray[np.float64]): y values.

    Returns:
        arrxy (NDArray[np.float64]): Subdivided array.
        x (NDArray[np.float64]): Subdivided x values.
        y (NDArray[np.float64]): Subdivided y values.
    """
    arrxy = np.repeat(arrxy, subdivisons, axis=0)
    arrxy = np.repeat(arrxy, subdivisons, axis=1)
    x = np.repeat(x, subdivisons)
    y = np.repeat(y, subdivisons)
    return arrxy, x, y


def radial_power(
    Pxy: arrfloat,
    fx: arrfloat,
    fy: arrfloat,
    subdivisions: int,
    n_bins: int,
) -> tuple[arrfloat, arrfloat]:
    """Get the radial bins for a power spectrum.

    Args:

    Returns:
        Pr (npt.NDArray[np.float64]): Power in each radial bin.
        r_edges (npt.NDArray[np.float64]): Edges of the radial bins.
    """
    r_func = lambda a, b: np.sqrt(a**2 + b**2)
    r_max = np.log10(r_func(fx.max(), fy.max()))
    r_min = np.log10(r_func(fx.min(), fy.min()))

    # subdivide Pxy by dividing each cell into subdivisions^2 cells
    # ie if subdivisions = 2, each cell will be divided into 4, double in x and in y
    # the subdivided cells will be filled with the same value as the original cell
    Pxy = subdivide_repeat(subdivisions, Pxy, fx, fy)[0]
    kx_s = np.linspace(fx.min(), fx.max(), fx.size * subdivisions)
    ky_s = np.linspace(fy.min(), fy.max(), fy.size * subdivisions)

    KX, KY = np.meshgrid(kx_s, ky_s)
    R = r_func(KX, KY).T
    r_edges = np.logspace(r_min, r_max, n_bins + 1)

    Pr = np.zeros(n_bins)
    for i in range(n_bins):
        mask: npt.NDArray[np.bool_] = (R >= r_edges[i]) & (R < r_edges[i + 1])
        if not mask.any():
            Pr[i] = np.nan
        else:
            Pr[i] = Pxy[mask].mean()
    return Pr, r_edges


def frame_power(
    cs: CenteredShock,
    chunk: int,
    t_idx: int,
    subdivisions: int,
    n_bins: int,
    centres: bool = True,
) -> tuple[arrfloat, arrfloat]:
    r"""Get the power spectrum of a frame.

    Args:
        cs (CenteredShock): CenteredShock object.
        chunk (int): Chunk number.
        t_idx (int): Time index.
        subdivisions (int): Number of subdivisions in each dimension.
        n_bins (int): Number of radial bins.
        centres (bool, optional): Whether to return the centres of the bins.
            Defaults to True.

    Returns:
        power (npt.NDArray[np.float64]): Power in each radial bin.
        kr (npt.NDArray[np.float64]): Radial bins.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from dotenv import dotenv_values
        >>> from tqdm import tqdm
        >>> import hybrid_jp as hj
        >>> import hybrid_jp.analysis as hja
        >>> if __name__ == "__main__":
        ...     # Environment variable DATA_DIR must be set to the directory containing
        ...     # the deck and sdf files
        ...     env = dotenv_values(".env")
        ...     data_dir = str(env["DATA_DIR"])
        ...     # Load deck and sdf files
        ...     deck = hja.load_deck(data_dir=data_dir)
        ...     SDFs, fpaths = hja.load_sdfs_para(
        ...         sdf_dir=data_dir,
        ...         dt=deck.output.dt_snapshot,
        ...         threads=7,
        ...         start=0,
        ...         stop=100,
        ...     )
        ...     # Unit conversions
        ...     for SDF in SDFs:
        ...         SDF.mag *= 1e9  # Convert to nT
        ...         SDF.mid_grid *= 1e-3  # Convert to km
        ...     # Create a CenteredShock object
        ...     cs = hja.CenteredShock(SDFs, deck)
        ...     # Set the number of chunks
        ...     N_CHUNKS = 10
        ...     cs.n_chunks = N_CHUNKS
        ...     # Get all frames in first downstream chunk
        ...     chunk = cs.downstream_start_chunk
        ...     all_frames = cs.valid_chunks[chunk, :]
        ...     valid_frames: hj.arrint = np.nonzero(all_frames)[0]
        ...     # Set fft properties
        ...     subdivisions = 8  # split each cell into (2^n)**2 subdivisions
        ...     num_radial_bins = 100  # These bins are logarithmically spaced
        ...     # Define containers for the power and radial bins
        ...     power = np.empty((valid_frames.size, num_radial_bins))
        ...     k = np.empty(num_radial_bins)
        ...     # Loop over all valid frames
        ...     for i, t_idx in tqdm(
        ...         enumerate(valid_frames),
        ...         total=valid_frames.size,
        ...         desc="Frame",
        ...     ):
        ...         power[i], k = hja.frame_power(
        ...             cs, chunk, t_idx, subdivisions, num_radial_bins
        ...         )
        ...     PSD = power.mean(axis=0)
        ...     fig, ax = plt.subplots()
        ...     ax.loglog(k, PSD, color="k", lw=1)
                ax.set_xlabel(r"$k\ [km^{-1}]$")
                ax.set_ylabel(r"$PSD(k)\ [nT^2\, km^{-2}]$")
        ...     plt.show()
    """

    def _func(sdf: SDF, qty: Literal["bx", "by", "bz"]) -> npt.NDArray[np.float64]:
        return getattr(sdf.mag, qty)

    xfunc = partial(_func, qty="bx")
    yfunc = partial(_func, qty="by")
    zfunc = partial(_func, qty="bz")
    cfuncs = [xfunc, yfunc, zfunc]

    p_components = np.empty((n_bins, 3))
    edges = np.empty(n_bins)
    for i, component in enumerate(cfuncs):
        data = cs.get_qty_in_frame(component, chunk, t_idx)[0]
        pxy, kx, ky = power_xy(data, cs.dx, cs.dy)
        pr, r_edges = radial_power(pxy, kx, ky, subdivisions, n_bins)
        p_components[:, i] = pr
        edges = r_edges

    if centres:
        kr = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), n_bins)
    else:
        kr = edges
    return p_components.sum(axis=1), kr
