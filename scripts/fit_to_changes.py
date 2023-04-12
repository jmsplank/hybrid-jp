"""Perform regressions on the shock changes."""
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import blue, green, red
from rich import print as rprint
from scipy.stats import linregress


def read_shock_changes_csv(path: Path) -> pd.DataFrame:
    """Read csv file containing shock changes.

    Args:
        path (Path): path to csv file.

    Returns:
        pd.DataFrame: dataframe containing shock changes.
    """
    df = pd.read_csv(path, index_col=0)
    return df


def regress(x, y) -> tuple[float, float]:
    """Perform a linear regression.

    Args:
        x (np.ndarray): x values.
        y (np.ndarray): y values.

    Returns:
        tuple[float, float]: slope and intercept.
    """
    reg = linregress(x, y)
    return reg.slope, reg.intercept  # type: ignore


def append_regress_to_dict(
    input_dict: dict[str, dict[str, float]], name: str, slope: float, intercept: float
) -> dict[str, dict[str, float]]:
    """Append a regression to a dictionary.

    Args:
        input_dict (dict[str, dict[str, float]]): dictionary to append to.
        name (str): name of the regression.
        slope (float): slope of the regression.
        intercept (float): intercept of the regression.

    Returns:
        dict[str, dict[str, float]]: dictionary with the regression appended.
    """
    input_dict[name] = dict(slope=slope, intercept=intercept)
    return input_dict


def plot_regress(
    ax: plt.Axes,
    label: str,
    t: np.ndarray,
    x: np.ndarray,
    slope: float,
    intercept: float,
    colour: str,
) -> None:
    """Plot a regression.

    - data shown as a scatterplot with `color`.
    - regression shown as a dashed line with `color`.

    Args:
        ax (plt.Axes): axis to plot on.
        label (str): label.
        t (np.ndarray): x values.
        x (np.ndarray): y values.
        slope (float): slope of the regression.
        intercept (float): intercept of the regression.
        colour (str): colour for regression line and data scatterplot.
    """
    ax.scatter(t, x, label=label, color=colour)
    fit_t = [t[0], t[-1]]
    fit_y = [t[0] * slope + intercept, t[-1] * slope + intercept]
    ax.plot(fit_t, fit_y, label=f"{label} fit", color=colour, linestyle="--")


def linear_regressions(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Perform linear regressions on the shock changes.

    df should have the following columns:
    - `shock`
    - `change_0`
    - `change_1`
    - `change_2`

    Args:
        df (pd.DataFrame): dataframe containing shock changes.

    Returns:
        dict[str, dict[str, float]]: dictionary containing regressions.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame(dict(shock=[1, 2, 3], change_0=[1, 2, 3], change_1=[1, 2, 3], change_2=[1, 2, 3]), index=[1, 2, 3])
        >>> linear_regressions(df)
        {
            'shock': {'slope': 0.0, 'intercept': 1.0},
            'change_0': {'slope': 0.0, 'intercept': 1.0},
            'change_1': {'slope': 0.0, 'intercept': 1.0},
            'change_2': {'slope': 0.0, 'intercept': 1.0}
        }

    """
    data: dict[str, dict[str, float]] = {}
    # regression for shock
    append_regress_to_dict(
        data,
        "shock",
        *regress(df.index, df.shock),
    )
    # regression for change_0
    append_regress_to_dict(
        data,
        "change_0",
        *regress(df.index, df.change_0),
    )
    # regression for change_1
    append_regress_to_dict(
        data,
        "change_1",
        *regress(df.index, df.change_1),
    )
    # regression for change_2
    append_regress_to_dict(
        data,
        "change_2",
        *regress(df.index, df.change_2),
    )
    return data


def plot_linear_regressions(
    data: dict[str, dict[str, float]], df: pd.DataFrame
) -> None:
    """Plot the linear regressions."""
    fig, ax = plt.subplots()
    colours = iter(["k", red, green, blue])
    for k, v in data.items():
        plot_regress(
            ax,
            k,
            np.array(df.index),
            np.array(df[k]),
            v["slope"],
            v["intercept"],
            next(colours),
        )
    plt.legend()
    ax.set_ylabel("x")
    ax.set_xlabel("t")
    plt.tight_layout()
    plt.show()


def mov_avg(data: np.ndarray, width: int) -> np.ndarray:
    """Perform a moving average.

    Args:
        data (np.ndarray): data to perform moving average on.
        width (int): width of moving average.

    Returns:
        np.ndarray: moving average of data. length is len(data) - width + 1.
    """
    return np.convolve(data, np.ones(width), "valid") / width


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


def moving_avg(df: pd.DataFrame) -> None:
    # Get the slope of each column in df
    lines = linear_regressions(df)
    width = 8  # width of moving average of form 2^n where n is an integer

    fig, ax = plt.subplots()

    changes = list(lines.keys())
    colours = iter(["k", red, green, blue])
    for change in changes:
        col = next(colours)
        slope = np.array(lines[change]["slope"] * df.index + lines[change]["intercept"])
        subtracted_slope = np.array(df[change] - slope)
        width_mov_avg = mov_avg(subtracted_slope, width)
        restored_slope = width_mov_avg + interpolate_to_midpoints(slope, width)

        # plot change as scatterplot of hollow circles
        ax.scatter(df.index, df[change], color=col, marker="o", facecolors="none")  # type: ignore
        # plot moving average of change as solid line
        ax.plot(
            interpolate_to_midpoints(np.array(df.index), width),
            restored_slope,
            color=col,
            linestyle="-",
        )
        ax.plot(
            interpolate_to_midpoints(np.array(df.index), width),
            mov_avg(np.array(df[change]), width),
            color=col,
            linestyle="--",
        )

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """Entry point."""
    path_shock_changes = Path("scripts/shock_changes.csv")
    df_shock_changes = read_shock_changes_csv(path_shock_changes)
    # plot_linear_regressions(linear_regressions(df_shock_changes), df_shock_changes)
    moving_avg(df_shock_changes)


if __name__ == "__main__":
    override_mpl.override("|krgb")
    main()
