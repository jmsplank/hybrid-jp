from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import numpy as np

from .change_points import binseg, find_shock_index_from_gradnd
from .dtypes import Grid
from .sdf_files import load_sdf_verified

SDF_Iterable = Callable[[Path], Any]


class SDFContainer(Protocol):
    @property
    def sdfs(self) -> Iterable:
        ...


@dataclass
class Shock:
    sdfs: list[Path]
    skip: int
    boundary_buffer: int
    grid: Grid
    mid_grid: Grid


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
        raise FileNotFoundError()
    if not folder.is_dir():
        raise Exception(f"{folder} is not a directory")

    # Get all SDF files in folder
    sdfs = sorted(folder.glob("*.sdf"), key=lambda p: int(p.stem))
    sdfs = sdfs[skip_at_start:]

    # Load first sdf to get extra metadata
    sdf0 = load_sdf_verified(sdfs[0])

    return Shock(
        sdfs=sdfs,
        skip=skip_at_start,
        boundary_buffer=0,
        grid=sdf0.grid,
        mid_grid=sdf0.mid_grid,
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


def para_iter_over_sdfs(shock: SDFContainer, fn: SDF_Iterable) -> list[Any]:
    def iter_fn() -> Iterator[Path]:
        for sdf in shock.sdfs:
            yield sdf

    with Pool() as pool:
        res = list(
            pool.imap(
                fn,
                iter_fn(),
            )
        )

    return res


def shock_and_changes(
    shock: SDFContainer,
    trim: slice,
):
    change_points = partial(
        get_change_points_x,
        trim_x=trim,
    )
    items: list[ChangePointsXResult] = para_iter_over_sdfs(shock, change_points)
    return ChangePoints(
        shock_index=[i.shock_index for i in items],
        change_points=[i.change_points for i in items],
    )
