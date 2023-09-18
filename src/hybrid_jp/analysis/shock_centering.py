from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from epoch_cheats.deck import Deck

from hybrid_jp.sdf_files import SDF


class NoTimestampError(Exception):
    ...


class NChunksNotSetError(Exception):
    """Raised in a CentredShock when n_chunks is not set."""

    message = "n_chunks must be set before calling this method."

    def __init__(self, message: str = message) -> None:
        self.message = message
        super().__init__(message)


class InvalidChunkError(Exception):
    """Raised when a chunk is not valid."""

    def __init__(self, chunk_idx: int, t_idx: int) -> None:
        self.message = f"Chunk {chunk_idx} at timestep {t_idx} is not valid."
        super().__init__(self.message)


class CenteredShock:
    def __init__(self, sdfs: list[SDF], deck: Deck) -> None:
        self.sdfs = sdfs
        self.deck = deck

        self.time = self._get_time()
        self.grid_km = self.sdfs[0].mid_grid * (1 / 1000)
        self._nd_median_y_cm = np.asarray(
            [np.median(sdf.numberdensity, axis=1) for sdf in sdfs]
        ).T / (100**3)

        self.shock_i = self.shock_index_from_nd()
        self.shock_x = self.grid_km.x[self.shock_i]

        self.dx = self.grid_km.x[1] - self.grid_km.x[0]
        self.dy = self.grid_km.y[1] - self.grid_km.y[0]

        self.dist_either_side = self._get_dist_either_side()
        self.max_widths = self.dist_either_side.max(axis=0)
        self.full_width = sum(self.max_widths)

        self.n_chunks: int | None = None

        # Can be set by set_chunk_i_missing
        self._chunks_info: dict[str, Any] = {}
        self._chunk_i: int | None = None
        self._missing: npt.NDArray[np.int_] | None = None
        self._downstream_start_chunk: int | None = None

        # Can be set by set_valid_chunks
        self._valid_chunks: npt.NDArray[np.bool_] | None = None

        # Can be set by set_start_offset
        self._start_offset: npt.NDArray[np.int64] | None = None

    def _get_chunks_info(self) -> dict[str, Any]:
        if not self._chunks_info:
            self._chunks_info = self.set_chunk_i_missing()
        return self._chunks_info

    @property
    def chunk_i(self) -> int:
        if self._chunk_i is None:
            ci = self._get_chunks_info()
            self._chunk_i = ci["chunk_i"]
        return self._chunk_i

    @property
    def missing(self) -> npt.NDArray[np.int_]:
        if self._missing is None:
            ci = self._get_chunks_info()
            self._missing = ci["missing"]
        return self._missing

    @property
    def downstream_start_chunk(self) -> int:
        if self._downstream_start_chunk is None:
            ci = self._get_chunks_info()
            self._downstream_start_chunk = ci["downstream_start_chunk"]
        return self._downstream_start_chunk

    @property
    def valid_chunks(self) -> npt.NDArray[np.bool_]:
        if self._valid_chunks is None:
            self._valid_chunks = self.set_valid_chunks()
        return self._valid_chunks

    @property
    def start_offset(self) -> npt.NDArray[np.int64]:
        if self._start_offset is None:
            self._start_offset = self.set_start_offset()
        return self._start_offset

    def _get_time(self):
        times = [sdf.tstamp for sdf in self.sdfs]
        if any(t is None for t in times):
            raise NoTimestampError("All SDFs require a tstamp that is not None.")
        return np.asarray(times)

    def shock_index_from_nd(self, threshold=0.37):
        mask = self._nd_median_y_cm > (self._nd_median_y_cm.max() * threshold)
        shock_i = np.asarray(np.argmax(mask, axis=0))
        return shock_i

    def _get_dist_either_side(self) -> npt.NDArray[np.float64]:
        """Number of elements in the array before & after the shock.

        Returns:
            dist_either_side: The distance from the shock in cells from the left
                and right boundaries
        """
        # Get the distance from the shock in cells from the left and right boundaries
        dist_either_side = np.asarray(
            [[s, abs(s - self.grid_km.x.size)] for s in self.shock_i]
        )

        return dist_either_side

    def reshape_qty_to_shock_arr(self, qty: npt.NDArray[np.float64]):
        """Reshape qty so that shock is aligned.

        Note:
            - qty must have shape (grid_km.x.size, time.size) i.e. the first dimension
              is the width of the grid in x and the second is the number of timesteps.

        Args:
            qty (npt.NDArray[np.float64]): The quantity to be reshaped.
        """
        required_shape = np.asarray([self.grid_km.x.size, self.time.size])
        if not np.array_equal(qty.shape, required_shape):
            raise ValueError(
                f"qty has shape {qty.shape} but required"
                f"shape is {tuple(required_shape)}"
            )
        arr = np.empty((self.full_width, self.time.size))
        arr[:] = np.nan

        for i in range(self.time.size):
            insert_i = self.max_widths[0] - self.dist_either_side[i][0]
            arr[insert_i : insert_i + self.grid_km.x.size, i] = qty[:, i]

        return arr

    def set_chunk_i_missing(self) -> dict[str, Any]:
        if self.n_chunks is None:
            raise NChunksNotSetError()

        # Needs to be 1 larger than the number of chunks
        chunk_grow = self.n_chunks + 1
        chunk_i: int = np.floor(self.full_width / chunk_grow).astype(int)
        ratio = self.max_widths / self.max_widths[1]
        fraction = ratio * (chunk_grow / ratio.sum())
        fraction = np.floor(fraction).astype(int)
        missing: npt.NDArray[np.int_] = self.max_widths - fraction * chunk_i

        self._chunk_i = chunk_i
        self._missing = missing
        self._downstream_start_chunk = fraction[0]

        return dict(
            chunk_i=chunk_i,
            missing=missing,
            downstream_start_chunk=fraction[0],
        )

    def set_valid_chunks(self):
        if self.n_chunks is None:
            raise NChunksNotSetError()

        # Reshape an array full of ones
        arr = self.reshape_qty_to_shock_arr(
            np.ones((self.grid_km.x.size, self.time.size))
        )
        arr[np.isnan(arr)] = 0  # Set areas of no data to False instead of NaN
        arr.astype(bool)  # arr is 1 where there would be data and 0 elsewhere.

        valid_chunks = np.zeros((self.n_chunks, self.time.size), dtype=bool)
        for i in range(self.time.size):
            valid = np.empty(self.n_chunks, dtype=bool)
            for j in range(self.n_chunks):
                cstart = self.chunk_i * j + self.missing[0]
                cend = cstart + self.chunk_i

                if cend > arr.shape[0]:
                    raise Exception(
                        f"End index for chunk {j} at timestep {i} is {cend} which is "
                        f"larger than the maximum size of {arr.shape[0]}"
                    )

                valid[j] = arr[cstart:cend, i].all()
            valid_chunks[:, i] = valid

        self._valid_chunks = valid_chunks

        return valid_chunks

    def set_start_offset(self) -> npt.NDArray[np.int64]:
        data_start = self.max_widths[0] - self.dist_either_side[:, 0]
        first_chunk = (
            np.argmax(self.valid_chunks, axis=0) * self.chunk_i + self.missing[0]
        )
        start_offset = (first_chunk - data_start).astype(np.int64)

        self._start_offset = start_offset

        return start_offset

    def get_x_offset_for_frame(self, chunk: int, t_idx: int) -> int:
        """The start index for a chunk at a timestep."""
        if not self.valid_chunks[chunk, t_idx]:
            raise InvalidChunkError(chunk, t_idx)
        data_chunk = (np.cumsum(self.valid_chunks, axis=0) - 1) * self.valid_chunks
        return data_chunk[chunk, t_idx] * self.chunk_i + self.start_offset[t_idx]

    def get_qty_in_frame(
        self,
        qty_func: Callable[[SDF], npt.NDArray[np.float64]],
        chunk: int,
        t_idx: int,
    ) -> tuple[npt.NDArray[np.float64], tuple[int, int]]:
        """Extract from an SDF using qty_func at a chosen chunk and timestep.

        Example:
            >>> def get_nd(sdf: SDF) -> npt.NDArray[np.float64]:
            ...     return sdf.numberdensity
            >>> qty, start_stop = cs.get_qty_in_frame(get_qty, 0, 0)

            >>> def get_median_bx(sdf: SDF) -> npt.NDArray[np.float64]:
            ...     return np.median(sdf.mag.bx, axis=1)
            >>> qty, start_stop = cs.get_qty_in_frame(get_median_bx, 0, 0)
            >>> plt.plot(x=cs.grid_km.x[slice(*start_stop)], y=qty)
        """
        if self.n_chunks is None:
            raise NChunksNotSetError()
        if 0 > chunk >= self.n_chunks:
            raise ValueError(f"chunk must be in range [0, {self.n_chunks}).")

        chunk_offset = self.get_x_offset_for_frame(chunk, t_idx)
        start_stop = (chunk_offset, chunk_offset + self.chunk_i)
        return qty_func(self.sdfs[t_idx])[slice(*start_stop)], start_stop
