from enum import Enum


class Label(Enum):
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
    if si:
        unit = si_units[var]
    else:
        unit = units[var]
    return f"{labels[var]} $[{unit}]$"
