from functools import partial
from multiprocessing import Pool, set_start_method
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm

from ..sdf_files import SDF, filefinder, load_sdf_verified


def load_sdfs_para(
    sdf_dir: str | Path,
    dt: float | None = None,
    threads: int | None = None,
    start: int | None = None,
    stop: int | None = None,
    verbose: bool = True,
    ipython: bool = False,
):
    if ipython:
        set_start_method("fork", force=True)

    if not isinstance(sdf_dir, Path):
        sdf_dir = Path(sdf_dir)
    files = list(Path(sdf_dir).glob("*.sdf"))
    n_files = len(files)
    _start = 0 if start is None else start
    _stop = n_files - 1 if stop is None else stop

    file_next = filefinder(sdf_dir, _start, _stop)
    SDFs: list[SDF] = []

    # Initialize progress bar as a no-op if verbose is False
    def slippery(iter: Iterable, *args, **kwargs) -> Iterable:
        return iter

    progress_bar = tqdm if verbose else slippery  # type: ignore

    load_sdf = partial(load_sdf_verified, dt=dt)
    with Pool(threads) as pool:
        SDFs = list(
            progress_bar(
                pool.imap(load_sdf, file_next),
                total=(_stop - _start + 1),
            )  # type: ignore
        )

    return SDFs, files
