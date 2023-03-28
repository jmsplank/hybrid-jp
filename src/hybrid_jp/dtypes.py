"""Base data types for the hybrid_jp."""
from collections import namedtuple

# Simulation grid x,y
Grid = namedtuple("Grid", ["x", "y"])

# Magnetic field components bx, by, bz
Mag = namedtuple("Mag", ["bx", "by", "bz"])
