from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from phdhelper.helpers import override_mpl

import hybrid_jp as hj

override_mpl.override("|krgb")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, header=0)


def df_to_rows_array(data: pd.DataFrame, cols: list[str]) -> np.ndarray:
    arr = np.empty((len(data), len(cols)))
    for i, col in enumerate(cols):
        arr[:, i] = data[col].values
    return arr


def rupture_algo(arr: np.ndarray, penalty: int = 20):
    print("Algorithm-ing")
    algo = rpt.Pelt(model="rbf").fit(arr)
    result = algo.predict(pen=penalty)
    print("Found some state transitions:")
    print(result)
    return result


def plot(
    data: pd.DataFrame,
    x_name: str,
    lines: list[float] | None = None,
    save_path: Path | None = None,
) -> None:
    cols = list(data)
    cols.remove(x_name)
    n_plots = len(cols)
    fig, axs = plt.subplots(n_plots, 1, sharex=True)

    for i in range(n_plots):
        axs[i].plot(data[x_name], data[cols[i]])

    if lines is not None:
        if len(data) in lines:
            lines.remove(len(data))
        for li in lines:
            for i in range(n_plots):
                axs[i].axvline(data[x_name].values[li], color="r", ls="--")  # type: ignore

    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    if save_path:
        axs[0].set_title(save_path.stem)
        plt.savefig(save_path, dpi=300)


def main(
    fnumber: int,
    save_name: str | None = None,
    save_path: Path = Path("scripts/temp/plots"),
):
    csv = Path(f"scripts/data/{fnumber:04d}.csv")
    data = load_csv(csv)
    print(data.head())
    arr = df_to_rows_array(
        data,
        [
            "Derived_Number_Density",
            "Magnetic_Field_Bx",
            "Magnetic_Field_By",
            "Magnetic_Field_Bz",
        ],
    )
    transitions = rupture_algo(arr)
    if save_name is None:
        save_name = f"i{fnumber:04d}_changes.png"
    plot(data, "Grid_Grid_mid", lines=transitions, save_path=save_path / save_name)


if __name__ == "__main__":
    for i in [10, 50, 100, 130, 150]:
        print(i)
        main(i)
