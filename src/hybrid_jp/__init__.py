"""Base module for hybrid_jp."""
from .arrays import df_to_rows_array, trim_vars
from .change_points import binseg
from .config import config_from_toml
from .dtypes import Grid, Mag, arrfloat, arrint
from .sdf_files import get_grid, get_mag, list_variables, print_variables
