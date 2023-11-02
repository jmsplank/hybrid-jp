"""Array operations for hybrid_jp."""
import numpy as np
import pandas as pd

from .dtypes import arrfloat, arrint


def create_orthonormal_basis_from_vec(
    v: arrfloat,
    e2: arrfloat | None = None,
    e2_plane: arrfloat | None = None,
) -> arrfloat:
    """create a basis of orthonormal vectors from an input vector `v`.

    The original components of the vector (i,j,k) in the original basis are transformed
    to a new orthonormal basis (e1,e2,e3) where the component (e1,0,0) in the new basis
    points in the direction of `v` in the original. Therefore, e2 and e3 lie in the
    plane normal to `v` and are mutually orthogonal.

    Args:
        v (arrfloat): 1D vector length 3 that is the e1 component
        e2 (arrfloat): 1D vector length 3 that is the e2 component
        e2_plane (arrfloat): 1D vec len 3, normal to plane that e2 must lie in.
            If present, overrides `e2`.

    Returns:
        arrfloat: square array shape (3,3). e_n=arr[n-1,:] where n=1,2,3

    Examples:
        >>> v = np.array([1, 2, 3], dtype=np.float64)
        >>> basis = create_orthonormal_basis_from_vec(v)
        >>> print("e1", basis[:, 0])
        >>> print("e2", basis[:, 1])
        >>> print("e3", basis[:, 2])
        >>> print("||e1||  ||e2||  ||e3||  ->  ", np.linalg.norm(basis, axis=1))
        >>> print("e1 . e2 = 0?", np.allclose(np.dot(orth[:, 0], orth[:, 1]), 0))
        >>> print("e1 . e3 = 0?", np.allclose(np.dot(orth[:, 0], orth[:, 2]), 0))
        >>> print("e2 . e3 = 0?", np.allclose(np.dot(orth[:, 1], orth[:, 2]), 0)
    """
    # v MUST be 1d
    assert len(v.shape) == 1
    # v MUST have shape 3
    assert v.size == 3

    # Make sure it's a unit vector
    e1 = v / np.linalg.norm(v)

    # 1. Generate a random vector e2
    #    e2 is NOT length 1 and is NOT orthogonal to `v`
    _e2: arrfloat
    if e2_plane is not None:
        assert e2_plane.size == 3
        e2_plane /= np.linalg.norm(e2_plane)
        _e2 = np.cross(e1, e2_plane)
    elif e2 is not None:
        _e2 = e2
    else:
        _e2 = np.random.random(3)

    _e2 /= np.linalg.norm(_e2)

    # 2. Project _e2 into a unit vector in the plane normal to `v`
    #    Done by subtracting the component parallel to `v` and normalising
    _e2 -= np.dot(_e2, e1) * e1
    _e2 /= np.linalg.norm(_e2)

    # 3. Get a third basis vector perpendicular to both v and _e2
    e3 = np.cross(e1, _e2)

    return np.stack((e1, _e2, e3), axis=0)


def rotate_arr_to_new_basis(basis: arrfloat, arr: arrfloat) -> arrfloat:
    """Rotate an array (nx, ny, 3) in i,j,k basis to a new `basis`.

    Args:
        basis (hj.arrfloat): (3,3) basis of (e1, e2, e3) where basis[0,:]=e1 etc.
        arr (hj.arrfloat): (nx, ny, 3) array of vectors on a grid (nx, ny)

    Returns:
        hj.arrfloat: (nx, ny, 3) array of vectors in e1,e2,e3 basis

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> scale = 100
        >>> xx, yy = np.mgrid[0:scale, 0:scale]
        >>> grid = np.empty((*xx.shape, 3))
        >>> grid[:, :, 0] = yy
        >>> grid[:, :, 1] = xx
        >>> # Add an oscillating part in k
        >>> grid[:, :, 2] = np.sin(2 * np.pi * xx / xx.max())
        >>> # e1=k in new basis
        >>> vec = np.array([0, 0, 1], dtype=np.float64)
        >>> print(f"{vec = }")
        >>> basis = create_orthonormal_basis_from_vec(vec)
        >>> print(f"{basis = }")
        >>> rot = rotate_arr_to_new_basis(basis, grid)
        >>> fig, axs = plt.subplots(2, 1)
        >>> # Plot i and j streamlines coloured by k
        >>> axs[0].streamplot(
        >>>     xx[:, 0],
        >>>     yy[0, :],
        >>>     grid[:, :, 0].T,
        >>>     grid[:, :, 1].T,
        >>>     density=2,
        >>>     arrowstyle="-",
        >>>     color=grid[:, :, 2].T,
        >>> )
        >>> # Visualise e1, should be a horizontal sine wave
        >>> axs[1].pcolormesh(xx[:, 0], yy[0, :], rot[:, :, 0].T)
        >>> axs[0].grid(False)
        >>> axs[1].grid(False)
        >>> fig.tight_layout()
        >>> plt.show()
    """
    # 1. Flatten the array into (3, nx*ny) where axis 0 is the ijk components
    arr_flat = arr.reshape(-1, 3).T
    # 2. Compute the dot product with the orthonormal basis
    new_basis_flat = np.dot(basis, arr_flat).T
    # 3. Reshape into oridinal dimension
    out = new_basis_flat.reshape(arr.shape)

    return out


def logspaced_edges(arr: arrfloat | arrint) -> arrfloat:
    """Expand a (possibly uneven but approximately) logarithmically spaced arr to edges.

    `arr` is shape (N,), therefore the returned array is shape (N+1,). The end points
    are given by n_0 - (n_1 - n_0)/2 and n_N + (n_N - n_{N-1})/2 for an array
    (n_0...n_N) = log10(arr). The spacing between values of the array is preserved, this
    is useful in the case of integer logspaced arrays where diff(log10(arr)) is not
    constant do to integer rounding. So, each value in the new array is half of the
    separation between the original values.

    Args:
        arr (arrfloat | arrint): Array of values.

    Returns:
        arrfloat: Array of edges.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> arr = np.unique(np.logspace(0, 2, 15, dtype=np.int32))
        >>> brr = logspaced_edges(arr)
        >>> fig, axs = plt.subplots(2, 1, figsize=(8, 2))
        >>> axlin, axlog = axs
        >>> orig = axlin.scatter(arr, np.zeros_like(arr), marker="x", color="k")
        >>> new = axlin.scatter(brr, np.zeros_like(brr), marker="+", color="r")
        >>> orig = axlog.scatter(arr, np.zeros_like(arr), marker="x", color="k")
        >>> new = axlog.scatter(brr, np.zeros_like(brr), marker="+", color="r")
        >>> axlog.set_xscale("log")
        >>> axlin.set_title("Linear scale")
        >>> axlog.set_title("Log scale")
        >>> axlin.set_yticks([])
        >>> axlog.set_yticks([])
        >>> axlog.set_xlabel("'x' = original, '+' = bin edges")
        >>> fig.tight_layout()
        >>> plt.show()
    """
    log_arr = np.log10(arr)  # log scale array
    log_diff = np.diff(log_arr)  # log difference

    # Add points on either side of log scaled array, equidistant
    # Original:     +....+....+..+....+....+..+....+....+
    # New:     +....+....+....+..+....+....+..+....+....+....+
    log_wide = np.asarray(
        [log_arr[0] - log_diff[0]] + log_arr.tolist() + [log_arr[-1] + log_diff[-1]]
    )
    log_wiff = np.diff(log_wide)  # Difference of longer array

    # Half of total difference between point i and i+2
    #        +....+....+....+..+....+....+..+....+....+....+
    # Diff:    4    4    4   2   4    4   2   4    4    4
    # Offset        4    4   4   2    4   4   2    4    4     4
    # 1/2 diff:     4    4   3   3    4   3   3    8    8
    log_diff = (log_wiff[:-1] + log_wiff[1:]) / 2

    # First point in new arr is half way between first two points in wide arr or
    # equivalently half of the difference between first and second points in original
    # arr behind the first point.
    first_point = (log_wide[0] + log_wide[1]) / 2
    lags_wide = np.ones(log_arr.size + 1) * first_point

    # Successive points created by adding the cumulative distance of that point from the
    # first point
    lags_wide[1:] = lags_wide[1:] + np.cumsum(log_diff)
    lags_wide = 10 ** (lags_wide)  # Rescale out of log space
    return lags_wide


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
