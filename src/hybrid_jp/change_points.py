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
    print("Binary Segmentation")
    algo = Binseg(model="l2", min_size=2).fit(arr)
    result = algo.predict(n_bkps=nseg)
    print("Found some state transitions:")
    print(result)
    return result
