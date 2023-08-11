from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from epoch_cheats import evaluate_deck, validate_deck
from epoch_cheats.deck import Deck
from joblib import Memory
from scipy.interpolate import interp1d  # type: ignore
from scipy.stats import linregress  # type: ignore
from tqdm import tqdm

from .arrays import interpolate_to_midpoints, mov_avg
from .change_points import binseg, find_shock_index_from_gradnd
from .dtypes import Grid
from .sdf_files import load_sdf_verified

memory = Memory("/tmp/cache", verbose=1)


def memory_clear():
    memory.clear()


@dataclass
class Shock:
    sdfs: list[Path]
    skip: int
    boundary_buffer: int
    grid: Grid
    mid_grid: Grid
    deck: Deck

    def __hash__(self) -> int:
        hasher = hashlib.sha256()
        paths = [i.stem for i in self.sdfs]
        for i in [*paths, self.skip, self.boundary_buffer]:
            hasher.update(str(i).encode())
        return int(hasher.hexdigest(), 16)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Shock):
            return False
        return hash(self) == hash(__value)

    @property
    def t(self) -> np.ndarray:
        dt = self.deck.output.dt_snapshot
        return np.arange(0, self.deck.control.t_end + dt, dt)


@dataclass
class ChangePointsXResult:
    shock_index: int
    change_points: list[int]


@dataclass
class ChangePoints:
    shock_index: list[int]
    change_points: list[list[int]]


def load(folder: Path, skip_at_start: int = 0) -> Shock:
    if not folder.exists():
        raise FileNotFoundError(f"{folder=} not found.")
    if not folder.is_dir():
        raise Exception(f"{folder} is not a directory")

    deck_path = folder / "input.deck"
    if not deck_path.exists():
        raise FileNotFoundError(f"{deck_path=} not found.")

    # Get all SDF files in folder
    sdfs = sorted(folder.glob("*.sdf"), key=lambda p: int(p.stem))
    sdfs = sdfs[skip_at_start:]

    # Load first sdf to get extra metadata
    sdf0 = load_sdf_verified(sdfs[0])

    deck = validate_deck(evaluate_deck(deck_path=deck_path))

    return Shock(
        sdfs=sdfs,
        skip=skip_at_start,
        boundary_buffer=0,
        grid=sdf0.grid,
        mid_grid=sdf0.mid_grid,
        deck=deck,
    )


def get_change_points_x(path_to_sdf: Path, trim_x: slice) -> ChangePointsXResult:
    sdf = load_sdf_verified(path_to_sdf)
    # Trim from end(s) of x, keep all y
    trimmed_nd = sdf.numberdensity[trim_x, :]
    # mean over y axis => len = x[trim]
    grad_nd_1d = np.gradient(trimmed_nd.mean(axis=1))

    # Use 1d gradnd
    shock_index = find_shock_index_from_gradnd(grad_nd_1d)
    # Use 2d nd
    change_points = binseg(trimmed_nd)

    return ChangePointsXResult(
        shock_index=shock_index,
        change_points=change_points,
    )


@memory.cache
def para_iter_over_sdfs(
    shock: Shock,
    trim: tuple[int | None, int | None],
) -> list[ChangePointsXResult]:
    def iter_fn() -> Iterator[Path]:
        for sdf in shock.sdfs:
            yield sdf

    change_points = partial(
        get_change_points_x,
        trim_x=slice(*trim),
    )
    with Pool() as pool:
        res = list(
            tqdm(
                pool.imap(
                    change_points,
                    iter_fn(),
                ),
                "Iterate over SDF",
                total=len(shock.sdfs),
            )
        )

    return res


@dataclass
class MovingAvgResult:
    t: np.ndarray
    shock: np.ndarray
    start_STR: np.ndarray
    end_STR: np.ndarray


def extrapolate_to_larger_arr(
    original_t: np.ndarray, changes: MovingAvgResult
) -> MovingAvgResult:
    """Remap the changes to the original time.

    Note:
        since bounds of original_t are necessarily outside of changes.t, the values are extrapolated.

    Args:
        original_t (np.ndarray): original time.
        changes (MovingAverageResult): changes to remap.

    Returns:
        MovingAverageResult: remapped changes.
    """
    out = []
    for change in [changes.start_STR, changes.end_STR, changes.shock]:
        f = interp1d(changes.t, change, kind="linear", fill_value="extrapolate")  # type: ignore
        out.append(f(original_t))

    remapped = MovingAvgResult(original_t, *out)
    return remapped


def moving_avg(shock: Shock, changes: ChangePoints) -> MovingAvgResult:
    # Linear regression for each changepoint group (1st, 2nd, 3rd)
    # Moving average of changepoint positions (- slope)
    # results in array shorter by `width`
    # => interpolate back to midpoints
    transposed_changes = np.array(
        [list(sl) for sl in list(zip(*changes.change_points))]
    )
    # Looks like
    # [[  25  245 1575 ...]     <- change_1
    #  [  60 1525 1580 ...]     <- change_2
    #  [  80 1540 1585 ...]]    <- change_3

    # Insert shock location at [0,:]
    # Looks like
    # [[ 530 1548 1589 ...]     <- shock
    #  [  25  245 1575 ...]     <- change_1
    #  [  60 1525 1580 ...]     <- change_2
    #  [  80 1540 1585 ...]]    <- change_3
    grouped_changes = np.stack([changes.shock_index, *transposed_changes], axis=0)
    n_changes = grouped_changes.shape[0]

    moving_avg_width = 8
    final_len = grouped_changes.shape[1] - moving_avg_width + 1
    res_arr = np.empty((n_changes, final_len))  # (changes, mov_avg)

    # Moving average for all the grouped_changes, width=width
    # NOTE: Resulting array is shorter by width-1
    for i in range(n_changes):
        change = grouped_changes[i, :]
        regress = linregress(shock.t, y=change)
        line = shock.t * regress.slope + regress.intercept  # type: ignore
        residuals = change - line
        avg = mov_avg(residuals, moving_avg_width)
        res_arr[i, :] = avg + interpolate_to_midpoints(line, moving_avg_width)

    # Create a container for averaged shock and STR locations
    t = interpolate_to_midpoints(shock.t, moving_avg_width)
    start_STR = np.min(res_arr, axis=0)
    end_STR = np.max(res_arr[:3, :], axis=0)
    shock_loc = res_arr[0, :]
    averaged_result = MovingAvgResult(
        t=t,
        start_STR=start_STR,
        end_STR=end_STR,
        shock=shock_loc,
    )
    # Stretch result back out to original size using linear extrapolation
    res = extrapolate_to_larger_arr(shock.t, averaged_result)

    return MovingAvgResult(shock.t, res.shock, res.start_STR, res.end_STR)


def shock_and_changes(
    shock: Shock,
    trim: slice,
):
    items: list[ChangePointsXResult] = para_iter_over_sdfs(
        shock,
        (trim.start, trim.stop),
    )
    return ChangePoints(
        shock_index=[i.shock_index for i in items],
        change_points=[i.change_points for i in items],
    )
