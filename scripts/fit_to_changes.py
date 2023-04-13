"""Perform regressions on the shock changes."""
from pathlib import Path
from typing import NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from phdhelper.helpers import override_mpl
from phdhelper.helpers.COLOURS import blue, green, red
from rich import print as rprint
from scipy.interpolate import interp1d
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


class MovingAverageResult(NamedTuple):
    """Holds the results of a moving average.

    Attributes:
        t (np.ndarray): time.
        start_STR (np.ndarray): x coordinate of the start of the STR.
        end_STR (np.ndarray): x coordinate of the end of the STR.
        shock (np.ndarray): x coordinate of the shock.
    """

    t: np.ndarray
    start_STR: np.ndarray
    end_STR: np.ndarray
    shock: np.ndarray

    def to_df(self) -> pd.DataFrame:
        """Convert to a dataframe.

        Returns:
            pd.DataFrame: dataframe containing t, start_STR, end_STR, shock.
        """
        return pd.DataFrame(
            dict(
                t=self.t,
                start_STR=self.start_STR,
                end_STR=self.end_STR,
                shock=self.shock,
            )
        )

    def save(self, path: Path) -> None:
        """Save to csv by converting to dataframe.

        Args:
            path (Path): path to save to.
        """
        self.to_df().to_csv(path, index=False)

    def _t_index(self, t: float) -> int:
        """Get the index of t in self.t.

        Args:
            t (float): time.

        Returns:
            int: index of t in self.t.
        """
        return int(np.argmin(np.abs(self.t - t)))

    def get_irow(self, index: int) -> tuple[float, tuple[float, float], float]:
        """Get the row at index.

        Args:
            index (int): index of row.

        Returns:
            tuple[float, tuple[float, float], float]: [t, (start_STR, end_STR), shock]
        """
        return (
            self.t[index],
            (self.start_STR[index], self.end_STR[index]),
            self.shock[index],
        )

    def get_row(self, t: float) -> tuple[tuple[float, float], float]:
        """Get the row at time t.

        Args:
            t (float): time.

        Returns:
            tuple[float, float, float]: [start_STR, end_STR, shock]
        """
        index = self._t_index(t)
        return self.get_irow(index)[1:]


def moving_avg(df: pd.DataFrame) -> MovingAverageResult:
    """Perform a moving average on the shock changes.

    df should have the following columns:
    - `shock`
    - `change_0`
    - `change_1`
    - `change_2`

    - Moving average performed on each column minus the linear regression of that column.
    - returned array is of length len(df) - width + 1 where width is the width of the moving average.
    - start coordinate of STR is minimum of the change points
    - end coordinate of STR is mean of change_1 and change_2
    - shock coordinate is maximum of shock and change_0

    Args:
        df (pd.DataFrame): dataframe containing shock changes.

    Returns:
        MovingAverageResult: moving average of shock changes.

    """
    # Get the slope of each column in df
    lines = linear_regressions(df)
    width = 8  # width of moving average of form 2^n where n is an integer

    changes = list(lines.keys())
    restored_slope = np.empty((len(changes), len(df.index) - width + 1))  # (x, t)
    for i, change in enumerate(changes):
        slope = np.array(lines[change]["slope"] * df.index + lines[change]["intercept"])
        subtracted_slope = np.array(df[change] - slope)
        width_mov_avg = mov_avg(subtracted_slope, width)
        restored_slope[i, :] = width_mov_avg + interpolate_to_midpoints(slope, width)

    new_t = interpolate_to_midpoints(np.array(df.index), width)
    str_start = np.min(restored_slope, axis=0)
    str_end = np.mean(restored_slope[2:, :], axis=0)
    shock = np.max(restored_slope[:2, :], axis=0)

    return MovingAverageResult(new_t, str_start, str_end, shock)


def plot_mov_avg(moving_avg: MovingAverageResult):
    """Plot the moving average.

    Args:
        moving_avg (MovingAverageResult): moving average to plot.
    """
    fig, ax = plt.subplots()
    ax.plot(moving_avg.t, moving_avg.shock, color="k", linestyle="-", label="Shock")
    ax.fill_between(
        moving_avg.t,
        moving_avg.start_STR,  # type: ignore
        moving_avg.end_STR,  # type: ignore
        color=green,
        alpha=0.5,
        label="STR",
    )

    ax.set_ylabel("x (m)")
    ax.set_xlabel("t (s)")

    plt.legend()
    plt.tight_layout()
    plt.show()


def remap_to_original_t(
    original_t: np.ndarray, changes: MovingAverageResult
) -> MovingAverageResult:
    """Remap the changes to the original time.

    Note:
        since bounds of original_t are necessaril outside of changes.t, the values are extrapolated.

    Args:
        original_t (np.ndarray): original time.
        changes (MovingAverageResult): changes to remap.

    Returns:
        MovingAverageResult: remapped changes.
    """
    out = []
    for change in [changes.start_STR, changes.end_STR, changes.shock]:
        f = interp1d(changes.t, change, kind="linear", fill_value="extrapolate")  # type: ignore
        out.append(f(original_t))

    remapped = MovingAverageResult(original_t, *out)
    return remapped


def main():
    """Entry point."""
    path_shock_changes = Path("scripts/shock_changes.csv")
    df_shock_changes = read_shock_changes_csv(path_shock_changes)
    # plot_linear_regressions(linear_regressions(df_shock_changes), df_shock_changes)
    mov_avg = moving_avg(df_shock_changes)
    remapped = remap_to_original_t(np.array(df_shock_changes.index), mov_avg)
    plot_mov_avg(remapped)
    remapped.save(Path("scripts/STR_locations.csv"))


if __name__ == "__main__":
    override_mpl.override("|krgb")
    main()
