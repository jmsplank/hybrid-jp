from typing import cast

import numpy as np

from .constants import VAR_BX, VAR_BY, VAR_BZ, VAR_GRID, VAR_GRID_MID
from .dtypes import Grid, Mag


def list_variables(
    data, show_name: bool = True, show_type: bool = False, show_size: bool = True
) -> list[list[str]]:
    dct = data.__dict__
    out = []
    for key in sorted(dct):
        try:
            val = dct[key]
            params = []
            if show_name:
                params.append(key)
            if show_type:
                params.append(type(val))
            if show_size:
                params.append(np.array2string(np.array(val.dims), separator=", "))
            out.append(params)

        except Exception:
            pass
    return out


def print_variables(
    data, show_name: bool = True, show_type: bool = False, show_size: bool = True
) -> None:
    vars = list_variables(data, show_name, show_type, show_size)
    for v in vars:
        print(" ".join(v))


def get_grid(data, mid: bool = False) -> Grid:
    attr_name = VAR_GRID if not mid else VAR_GRID_MID
    attr = getattr(data, attr_name).data
    return Grid(x=attr[0], y=attr[1])


def get_mag(data) -> Mag:
    components = []
    for i in [VAR_BX, VAR_BY, VAR_BZ]:
        components.append(getattr(data, i).data)
    return Mag._make(components)
