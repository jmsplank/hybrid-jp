"""Script for plotting spectra.

1. Load the csv containing STR start and end x coordinates for each time step
2. match each time step in data to an sdf file
3. for each time step
    1. load data file
    2. read magnetic field data
    3. split timeseries into upstream, str, downstream based on csv file
    4. calculate spectrum of each region
    5. for each region spectrum
        1. split into inertial & ion range if possible
        2. fit slope (spectral index) to each range
    5. return k, PSD, inertial slope and ion slope for each region

plot a - overview of regions:
1. plot a parameter e.g. numberdensity on pcolormesh
2. overplot str region

plot b - example of single time step:
1. choose a time step, maybe 0128
2. plot upstream, str, and downstream power spectrum (k vs PSD)
3. overplot slopes for each region

plot c - region spectra for all time:
1. average together the spectra for every time step
    1. account for uneven length regions
    2. possibly also account for offest k bins
2. plot of avg US, STR, DS power spectra
3. overplot avg slopes for each region

plot d - spectral index evolution:
1. plot spectral index as function of time for each range (MHD, ion) for each region

"""
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from epoch_cheats import evaluate_deck
from fit_to_changes import MovingAverageResult
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import blue, green, red
from rich import print
from scipy.signal import periodogram
from scipy.stats import binned_statistic
from sdf import BlockList
from tqdm import tqdm

import hybrid_jp as hj


def load_locations(path_to_csv: Path) -> MovingAverageResult:
    """Load the STR_locations csv into a MovingAverageResult object.

    Args:
        path_to_csv (Path): Path to the csv file.

    Returns:
        MovingAverageResult: A MovingAverageResult object containing data from csv.
    """
    df = pd.read_csv(path_to_csv)
    return MovingAverageResult(
        **{str(k): np.array(v) for k, v in df.to_dict(orient="list").items()}
    )


def load_data(path_to_sdf: Path) -> BlockList:
    """Load data from sdf file.

    Args:
        path_to_sdf (Path): Path to sdf file.

    Returns:
        BlockList: Data from sdf file.
    """
    return hj.sdf_files.load(path_to_sdf)


def make_t_arr(dt: float, t_end: float, t_start: float = 0.0) -> np.ndarray:
    """Make an array of time values.

    Args:
        dt (float): Time step.
        t_end (float): End time.
        t_start (float, optional): Start time. Defaults to 0.0.

    Returns:
        np.ndarray: Array of time values.
    """
    return np.arange(t_start, t_end, dt)


def get_index(arr: np.ndarray, value: float, axis: int = 0) -> int:
    """Get index of value in array.

    Args:
        arr (np.ndarray): Array to search.
        value (float): Value to search for.
        axis (int, optional): Axis to search along. Defaults to 0.

    Returns:
        int: Index of value.
    """
    return int(np.argmin(np.abs(arr - value), axis=axis))


def get_PSD_of_component(component: np.ndarray, dt: float):
    """Calculate the power spectrum of a component of the magnetic field.

    - uses periodogram from scipy.signal with arguments:
        - detrend="constant"
        - window="hann"
        - return_onesided=True
        - scaling="density"

    Args:
        component (np.ndarray): Component of magnetic field.
        dt (float): Time step.

    Returns:
        tuple[np.ndarray, np.ndarray]: Frequency and power spectrum.
    """
    f, Pxx = periodogram(
        x=component,
        fs=1 / dt,
        detrend="constant",
        window="hann",
        return_onesided=True,
        scaling="density",
    )
    return f, Pxx


class RegionPSD(NamedTuple):
    """Holds PSD data for a region.

    Attributes:
        f (np.ndarray): Frequency.
        PSD (np.ndarray): Power spectrum.
        inertial_slope (float, optional): Inertial range slope. Defaults to None.
        ion_slope (float, optional): Ion range slope. Defaults to None.
    """

    f: np.ndarray
    PSD: np.ndarray
    inertial_slope: float | None = None
    ion_slope: float | None = None


class TimestepPSD(NamedTuple):
    """Holds region PSD data for a single time step.

    Attributes:
        US (RegionPSD): Upstream region.
        STR (RegionPSD): Shock transition region.
        DS (RegionPSD): Downstream region.
    """

    US: RegionPSD
    STR: RegionPSD
    DS: RegionPSD


def timestep(path_to_sdf: Path, STR_x: tuple[float, float], dt: float) -> TimestepPSD:
    """Calculate the power spectrum for a single time step.

    Args:
        path_to_sdf (Path): Path to sdf file.
        STR_x (tuple[float, float]): Start and end x coordinates of STR region.
        dt (float): Time step.

    Returns:
        TimestepPSD: PSD data for each region.
    """
    data = load_data(path_to_sdf)
    mag = hj.sdf_files.get_mag(data)
    mag = mag.mean_over_y()

    grid = hj.sdf_files.get_grid(data, mid=True)
    x = grid.x

    # get index in x array of STR region
    STR_i = (get_index(x, STR_x[0]), get_index(x, STR_x[1]))

    # list of region tuples
    regions: list[tuple[int, int]] = [(0, STR_i[0]), STR_i, (STR_i[1], len(x))]
    region_names = ("US", "STR", "DS")

    out = {}
    for region, name in zip(regions, region_names):
        # for each component of mag, trim to index of region
        # stack like (3, len(x))
        mag_components = np.stack([m[region[0] : region[1]] for m in mag], axis=0)
        pwr = []
        f: np.ndarray | None = None

        # Iterate over each component to get power
        for i in range(mag_components.shape[0]):
            f, Pxx = get_PSD_of_component(mag_components[i, :], dt)

            # Makes the linter happy
            if f is None:
                raise ValueError("f is None")

            # We only want freq>0 because of log scaling later
            Pxx = Pxx[f > 0]
            f = f[f > 0]
            pwr.append(Pxx)

        # Makes the linter happy
        if f is None:
            raise ValueError("f is None")

        # Average across components
        PSD = np.mean(pwr, axis=0)
        out[name] = RegionPSD(f, PSD)

    return TimestepPSD(**out)


def get_nd_for_each_timestep(sdf_names: list[Path]) -> np.ndarray:
    """Loop over sdf files and get <nd(x)>_y for each timestep.

    Args:
        sdf_names (list[Path]): List of paths to sdf files.

    Returns:
        np.ndarray: Array of <nd(x)>_y for each timestep.
    """
    res = []
    for sdf in tqdm(sdf_names):
        data = hj.sdf_files.load(sdf)
        nd = data.Derived_Number_Density.data
        res.append(np.mean(nd, axis=1))
    return np.array(res)


def plota(sdf_names: list[Path], locations: MovingAverageResult, dt: float):
    """Plot of STR over the average numberdensity."""
    times = locations.t
    times_edges = np.linspace(times[0] - dt / 2, times[-1] + dt / 2, len(times) + 1)
    x = get_x(mid=False)
    # x_mid = get_x(mid=True)
    nd_timesteps = get_nd_for_each_timestep(sdf_names)

    XX, TT = np.meshgrid(x, times_edges)

    fig, ax = plt.subplots()
    im = ax.pcolormesh(XX, TT, nd_timesteps)
    plt.colorbar(im, label="n ($m^{-3}$)")

    ax.fill_betweenx(
        times,
        locations.start_STR,  # type: ignore
        locations.end_STR,  # type: ignore
        edgecolor="k",
        facecolor="#00000033",
        label="STR",
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("t (s)")
    plt.tight_layout()
    plt.savefig("scripts/temp/make_regions/plota.png", dpi=300)
    plt.show()
    del fig, ax


def plotb(data: TimestepPSD, fname: Path):
    """Plot of PSD for each region of a single timestep."""
    fig, ax = plt.subplots()
    for name, region in data._asdict().items():
        ax.loglog(region.f, region.PSD, label=name)
    ax.legend()
    ax.set_xlabel("f")
    ax.set_ylabel("PSD")
    ax.set_title(f"Power Spectra of Upstream, STR, Downstream Regions for {fname.stem}")
    plt.tight_layout()
    plt.savefig("scripts/temp/make_regions/plotb.png", dpi=300)
    plt.show()
    del fig, ax


def plotc(data: list[TimestepPSD]):
    """Plot of average PSD in each region over all time steps."""
    fig, ax = plt.subplots()
    res: dict[str, list[np.ndarray]] = dict(US=[], STR=[], DS=[])
    for step in data:
        for name, region in step._asdict().items():
            res[name].append(np.stack([region.f, region.PSD], axis=0))

    res2 = {k: np.concatenate(v, axis=1) for k, v in res.items()}
    lower_bin = np.log10(np.min([np.min(v[0, :]) for v in res2.values()]))
    upper_bin = np.log10(np.max([np.max(v[0, :]) for v in res2.values()]))
    bin_mids_x = np.logspace(lower_bin, upper_bin, 100)
    delta_f = np.log10(bin_mids_x[1]) - np.log10(bin_mids_x[0])
    bin_edges_x = np.logspace(
        lower_bin - delta_f / 2,
        upper_bin + delta_f / 2,
        len(bin_mids_x) + 1,
    )

    colours = iter([red, green, blue])
    for name, region in res2.items():
        col = next(colours)
        bin_res = binned_statistic(
            x=region[0, :],
            values=region[1, :],
            statistic="mean",
            bins=bin_edges_x,  # type: ignore
        ).statistic

        ax.loglog(bin_mids_x, bin_res, label=name, color=col)

    ax.legend()
    ax.set_xlabel("f")
    ax.set_ylabel("PSD")
    ax.set_title("Average Power Spectra of Upstream, STR, Downstream Regions")
    plt.tight_layout()
    plt.savefig("scripts/temp/make_regions/plotc.png", dpi=300)
    plt.show()
    del fig, ax


def get_x(mid=False) -> np.ndarray:
    """Open U6T40/0000.sdf and get x or x_mid.

    Args:
        mid (bool, optional): Get edges or midopints. Defaults to False.

    Returns:
        np.ndarray: Array of x grid edges or midpoints
    """
    data = hj.sdf_files.load(Path("U6T40/0000.sdf"))
    grid = hj.sdf_files.get_grid(data, mid=mid)
    return grid.x


def main(path_to_csv: Path):
    """Entry point."""
    override_mpl.override()
    locations = load_locations(path_to_csv)
    deck = evaluate_deck(Path("U6T40/input.deck"))

    dt = float(deck["output"]["dt_snapshot"])  # type: ignore
    t = make_t_arr(dt, deck["control"]["t_end"])  # type: ignore
    start_index = np.argmin(np.abs(t - locations.t[0]))
    end_index = len(t) + 1
    indices = range(start_index, end_index)
    sdf_names = [Path(f"U6T40/{i:04d}.sdf") for i in indices]

    power_timesteps: list[TimestepPSD] = []
    for i, sdf_name in enumerate(sdf_names):
        print(f"Timestep {i} of {len(sdf_names)}")
        i_t, i_STR, i_shock = locations.get_irow(i)
        power_timesteps.append(timestep(sdf_name, i_STR, dt))
        print()

    plota(sdf_names, locations, dt)

    # example plot
    # n = 128
    # plotb(power_timesteps[n], fname=sdf_names[n])

    # Plot of means
    # plotc(power_timesteps)


if __name__ == "__main__":
    csv_location = Path("scripts/STR_locations.csv")
    main(csv_location)
