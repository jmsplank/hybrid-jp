from functools import partial

import numpy as np

from .. import arrfloat, sdf_files
from . import CenteredShock


class MagneticField:
    """Use a `CenteredShock` to get shock-aligned magnetic fields."""

    def __init__(self, cs: CenteredShock) -> None:
        """init

        Args:
            cs (CenteredShock): `cs.n_chunks` must be set.
        """
        self.cs = cs
        self.chunk_i = self.cs.chunk_i
        self.ny = self.cs.grid.y.size

    @staticmethod
    def get_mag_xyz(sdf: sdf_files.SDF, q: str) -> arrfloat:
        """Getter for component `q=bx,by,bz` of sdf.Mag.

        Args:
            sdf (sdf_files.SDF): sdf object
            q (str): the magnetic field component, one of `bz,by.bz`

        Returns:
            arrfloat: array (nx,ny) of b field in the grid
        """
        return getattr(sdf.mag, q)

    # Shortcut functions for getting any of the 3 field components
    f_x = partial(get_mag_xyz, q="bx")
    f_y = partial(get_mag_xyz, q="by")
    f_z = partial(get_mag_xyz, q="bz")
    # Compose them in a list for iterating
    funcs = [f_x, f_y, f_z]

    def b_xyz_frame(
        self,
        chunk: int,
        t_idx: int,
    ) -> arrfloat:
        """Given a chunk number (<`cs.n_chunks`) and a timestamp, return b_xyz.

        Args:
            chunk (int): chunk number must be `0<=chunk<cs.n_chunks`
            t_idx (int): timestamp must be one of indices where
                `cs.valid_chunks[chunk]==1`

        Returns:
            arrfloat: (cs.chunk_i, ny, 3)
        """
        out = np.empty((self.chunk_i, self.ny, 3), dtype=np.float64)
        for i, f in enumerate(self.funcs):
            out[:, :, i], _ = self.cs.get_qty_in_frame(
                qty_func=f, chunk=chunk, t_idx=t_idx
            )
        return out

    def b_xyz_chunk(self, chunk: int) -> arrfloat:
        """get b_xyz frame for every valid timestep in chunk

        Note:
            The size of axis 0 (number of timesteps) can be found from
            cs.valid_chunks[chunk].sum().

        Args:
            chunk (int): must be `0<=chunk<cs.n_chunks`

        Returns:
            arrfloat: (n_times, cs.chunk_i, ny, 3)
        """
        times = np.nonzero(self.cs.valid_chunks[chunk])[0]
        out = np.empty((times.size, self.chunk_i, self.ny, 3), dtype=np.float64)
        for i, time in enumerate(times):
            out[i] = self.b_xyz_frame(chunk=chunk, t_idx=time)

        return out
