"""Base module for hybrid_jp."""
from .arrays import df_to_rows_array, trim_vars
from .change_points import binseg
from .dtypes import Grid, Mag
from .sdf_files import get_grid, get_mag, list_variables, print_variables
