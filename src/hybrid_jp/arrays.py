"""Array operations for hybrid_jp."""
import numpy as np
import pandas as pd


def trim_var(var: np.ndarray, slc: slice) -> np.ndarray:
    """Trim a variable to a slice.

    Args:
        var (np.ndarray): Variable to trim.
        slc (slice): Slice to trim to.

    Returns:
        np.ndarray: Trimmed variable.
    """
    return var[slc]


def trim_vars(vars_list: list[np.ndarray], slc: slice) -> list[np.ndarray]:
    """Trim a list of variables to a slice.

    Args:
        vars_list (list[np.ndarray]): List of variables to trim.
        slc (slice): Slice to trim to.

    Returns:
        list[np.ndarray]: Trimmed variables.

    Example:
        >>> trim_vars([np.arange(10), np.arange(10)], slice(0, 5))
        [array([0, 1, 2, 3, 4]), array([0, 1, 2, 3, 4])]
    """
    return [trim_var(var, slc) for var in vars_list]


def df_to_rows_array(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Convert dataframe into 2d np array [x, columns].

    Args:
        data (pd.DataFrame): data
        cols (list[str]): Column names to transform into array

    Returns:
        np.ndarray: 2d array of data
    """
    arr = np.empty((len(data), len(cols)))
    for i, col in enumerate(cols):
        arr[:, i] = data[col].values
    return arr


def interpolate_to_midpoints(arr: np.ndarray, width: int) -> np.ndarray:
    """Interpolate to midpoints.

    returns an array of length len(arr) - width + 1.

    Args:
        arr (np.ndarray): array to interpolate.
        width (int): width of moving average.

    Returns:
        np.ndarray: interpolated array.
    """
    return np.linspace(arr[0], arr[-1], len(arr) - width + 1)


def mov_avg(data: np.ndarray, width: int) -> np.ndarray:
    """Perform a moving average.

    Args:
        data (np.ndarray): data to perform moving average on.
        width (int): width of moving average.

    Returns:
        np.ndarray: moving average of data. length is len(data) - width + 1.
    """
    return np.convolve(data, np.ones(width), "valid") / width
