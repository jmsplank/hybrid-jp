import tomllib
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, DirectoryPath, PositiveInt, RootModel


class ShockParams(BaseModel):
    test_data_dir: DirectoryPath
    data_dir: Path
    n_chunks: PositiveInt
    n_threads: PositiveInt
    start_sdf: PositiveInt
    stop_sdf: PositiveInt


class Shocks(RootModel):
    root: Dict[str, ShockParams]

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]

    def __getattr__(self, attr):
        return self.root[attr]


class Config(BaseModel):
    shocks: Shocks
    use_shock: str

    @property
    def shk(self):
        shock_name = self.use_shock
        if shock_name not in self.shocks:
            raise KeyError(f"use_shock={shock_name} is not a valid shock.")
        return self.shocks[shock_name]


class TOMLConfigLoader:
    def __init__(self, config_toml_path: str | Path) -> None:
        if not isinstance(config_toml_path, Path):
            self.config_toml_path = Path(config_toml_path)
        else:
            self.config_toml_path = config_toml_path

    @staticmethod
    def _broadcast_defaults(attrs: dict[str, Any]) -> dict[str, Any]:
        data = attrs.copy()
        defaults = data.pop("default")
        for shock_name, shock_params in data.copy().items():
            for parameter_name, parameter_value in defaults.items():
                if parameter_name not in shock_params:
                    data[shock_name][parameter_name] = parameter_value
        return data

    def load(self) -> dict[str, Any]:
        with open(self.config_toml_path, "rb") as file:
            data = tomllib.load(file)

        data["shocks"] = self._broadcast_defaults(data["shocks"])

        return data


def get_config_from_toml(toml_path: Path | str) -> Config:
    loader = TOMLConfigLoader(toml_path)
    return Config(**loader.load())
