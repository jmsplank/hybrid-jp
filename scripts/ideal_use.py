"""Ideal use case for the hybrid_jp module.

23/06/12


"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from rich import print

import hybrid_jp.shock as shk


def main():
    # Load the data from a folder
    # Folder contains
    # U6T40/
    #     0001.sdf
    #     0002.sdf
    #     0003.sdf
    #     ... .sdf
    #     input.deck
    data_folder = Path("U6T40")
    data = shk.load(data_folder)

    # Detect changepoints in data
    # Trim off last 10 x indices
    shock_and_changes = shk.shock_and_changes(data, slice(None, -10))
    print(shock_and_changes.shock_index)


if __name__ == "__main__":
    main()
