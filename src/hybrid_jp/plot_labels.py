"""Labels to use when plotting variables."""
from enum import Enum


class Label(Enum):
    """Variable name labels."""

    ND = "ND"
    X = "X"


labels = {
    Label.ND: "Number density",
    Label.X: "$x$",
}
units = {
    Label.ND: "cm^{-3}",
    Label.X: "m",
}
si_units = {
    Label.ND: "m^{-3}",
    Label.X: "m",
}


def make_label(var: Label, si: bool = True) -> str:
    """Make a label for a variable.

    Args:
        var (Label): Variable to make a label for.
        si (bool, optional): Whether to use SI units. Defaults to True.

    Returns:
        str: Label for the variable.
    """
    if si:
        unit = si_units[var]
    else:
        unit = units[var]
    return f"{labels[var]} $[{unit}]$"
