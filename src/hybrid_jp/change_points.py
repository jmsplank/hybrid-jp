"""Chagne point detection using ruptures library."""
import numpy as np
from ruptures import Binseg


def binseg(arr: np.ndarray, nseg=3) -> list[int]:
    """Use Binary Segmentation algorithm to detect change points.

    Args:
        arr (np.ndarray): 2d array of [x,features]
        nseg (int, optional): number of change points to return. Defaults to 3.

    Returns:
        list[int]: index of change points
    """
    algo = Binseg(model="l2", min_size=2).fit(arr)
    result = algo.predict(n_bkps=nseg)

    # remove element equal to len of first dimension of arr if exists
    if result[-1] == arr.shape[0]:
        result = result[:-1]
    return result  # type: ignore


def find_shock_index_from_gradnd(gradnd: np.ndarray) -> int:
    """Find the index of the shock from gradient of the number density.

    - If gradnd is 2d then find the peak of the mean.
    - Assume the shock is located at the peak of grad nd

    Args:
        gradnd (np.ndarray): gradient of the number density.

    Returns:
        int: index of the shock.
    """
    if gradnd.ndim == 2:
        gradnd = gradnd.mean(axis=1)

    return int(np.argmax(gradnd))
