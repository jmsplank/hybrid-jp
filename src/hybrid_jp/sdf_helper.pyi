# flake8: noqa
"""Defines type hints for the sdf_helper module."""
import sdf

def getdata(
    fname: str, wkd: str | None = None, verbose: bool = True, squeeze: bool = False
) -> sdf.BlockList: ...
def list_variables(data: sdf.BlockList) -> None: ...
