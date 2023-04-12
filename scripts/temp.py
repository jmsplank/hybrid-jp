"""This script is used to remap the shock changes data to x values."""
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print


class Data(ABC):
    """Abstract base class for data."""

    @abstractmethod
    def get_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas DataFrame."""
        pass


class StringData(Data):
    """Data from a string.

    Attributes:
        data (str): The data as a string.
    """

    def __init__(self, data: str):
        """Initialize the StringData instance.

        Args:
            data (str): The data as a string.
        """
        self._data = data
        self._df = self._read_string_data()

    def _read_string_data(self) -> pd.DataFrame:
        return pd.read_csv(StringIO(self._data), index_col="t")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas DataFrame."""
        return self._df


class CsvData(Data):
    """Data from a csv file.

    Attributes:
        path (Path): The path to the csv file.
    """

    def __init__(self, path_to_data: Path) -> None:
        """Initialize the CsvData instance.

        Args:
            path_to_data (Path): The path to the csv file.
        """
        self.path: Path = path_to_data
        self._df = self._read_csv_data()

    def _read_csv_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path, index_col="t")

    def get_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas DataFrame."""
        return self._df


class DataFactory:
    """Factory class for creating Data instances."""

    @staticmethod
    def create_data_from_csv(data: str | Path) -> CsvData:
        """Create a CsvData instance from a csv file.

        Args:
            data (str | Path): The path to the csv file.

        Returns:
            CsvData: The CsvData instance.
        """
        if not isinstance(data, Path):
            data = Path(data)
        return CsvData(data)

    @staticmethod
    def create_data_from_string(data: str) -> StringData:
        """Create a StringData instance from a string.

        Args:
            data (str): The data as a string.

        Returns:
            StringData: The StringData instance.

        Example:
            >>> data = '''
            ... t,shock,change_0,change_1,change_2
            ... 6.549165220934418,16772013.45177251,16728842.24597644,16782806.253221527,16890734.2677117
            ... 7.204081743027859,16707256.643078407,16512986.2169961,16674878.238731356,16836770.260466613
            ... 7.858998265121301,16664085.437282339,16405058.20250593,16620914.23148627,16782806.253221527
            ... 8.513914787214743,16588535.82713922,16405058.20250593,16566950.224241186,16998662.282201868
            ... 9.168831309308185,16545364.62134315,16351094.195260845,16512986.2169961,16674878.238731356
            ... 9.823747831401626,16480607.812649049,16351094.195260845,16459022.209751016,16620914.23148627
            ... 10.478664353495068,16448229.408301998,16297130.188015759,16405058.20250593,16566950.224241186
            ... 11.13358087558851,16361886.99670986,16297130.188015759,16351094.195260845,16459022.209751016
            ... 11.788497397681951,16307922.989464777,16243166.180770673,16297130.188015759,16459022.209751016
            ... '''
            >>> csv_data = DataFactory.create_data_from_string(data)
            >>> csv_data.get_dataframe()
                        shock      change_0      change_1      change_2
            t
            6.549165    1.677201e+07  1.672884e+07  1.678281e+07  1.689073e+07
            7.204082    1.670726e+07  1.651299e+07  1.667488e+07  1.683677e+07
            7.858998    1.666409e+07  1.640506e+07  1.662091e+07  1.678281e+07
            8.513915    1.658854e+07  1.640506e+07  1.656695e+07  1.699866e+07
            9.168831    1.654536e+07  1.635109e+07  1.651299e+07  1.667488e+07
        """
        return StringData(data)


class CsvDataRemapper:
    """Remap data from a csv file to a new x range."""

    def __init__(self, data: Data, x_start: float, x_end: float, x_count: int):
        self._data = data
        self._x = self._create_x_values(x_start, x_end, x_count)

    def _create_x_values(
        self, x_start: float, x_end: float, x_count: int
    ) -> np.ndarray:
        return np.linspace(x_start, x_end, x_count)

    def _find_closest_x(self, target_value):
        return (np.abs(self._x - target_value)).argmin()

    def remap_to_x(self) -> pd.DataFrame:
        df = self._data.get_dataframe()

        # Create the x values array
        t = df.index.values

        # Map the data using the x values
        remapped_data = pd.DataFrame(index=self._x, columns=df.columns)

        for column in [str(c) for c in df.columns.tolist()]:
            indices = df[column].map(self._find_closest_x)

        return remapped_data


# Usage
data = """
t,shock,change_0,change_1,change_2
6.549165220934418,16772013.45177251,16728842.24597644,16782806.253221527,16890734.2677117
7.204081743027859,16707256.643078407,16512986.2169961,16674878.238731356,16836770.260466613
7.858998265121301,16664085.437282339,16405058.20250593,16620914.23148627,16782806.253221527
8.513914787214743,16588535.82713922,16405058.20250593,16566950.224241186,16998662.282201868
9.168831309308185,16545364.62134315,16351094.195260845,16512986.2169961,16674878.238731356
9.823747831401626,16480607.812649049,16351094.195260845,16459022.209751016,16620914.23148627
10.478664353495068,16448229.408301998,16297130.188015759,16405058.20250593,16566950.224241186
11.13358087558851,16361886.99670986,16297130.188015759,16351094.195260845,16459022.209751016
11.788497397681951,16307922.989464777,16243166.180770673,16297130.188015759,16459022.209751016
"""

x_start = 5396.4007245085295
x_end = 17263085.917702787
x_count = 1600

# Create CsvData instance using the factory method
# csv_data = DataFactory.create_data_from_string(data)
csv_data = DataFactory.create_data_from_csv(Path("scripts/shock_changes.csv"))
print(csv_data.get_dataframe())
# Initialize CsvDataRemapper with the CsvData instance
remapper = CsvDataRemapper(csv_data, x_start, x_end, x_count)

# Remap the data to x
remapped_df = remapper.remap_to_x()

print(remapped_df)
