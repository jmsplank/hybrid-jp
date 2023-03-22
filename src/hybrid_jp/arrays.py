import numpy as np


def trim_var(var: np.ndarray, slc: slice):
    return var[slc]


def trim_vars(vars_list: list[np.ndarray], slc: slice):
    return [trim_var(var, slc) for var in vars_list]
