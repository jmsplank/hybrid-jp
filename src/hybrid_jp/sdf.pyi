# flake8: noqa
"""Defines type hints for the sdf module."""
from abc import ABC

from numpy import ndarray

class Block:
    data_length: int
    datatype: type
    data: tuple[ndarray, ndarray] | ndarray
    dims: tuple
    id: str
    name: str

class BlockArray(Block): ...

class BlockMesh(Block):
    extents: tuple
    geometry: int
    data: tuple[ndarray, ndarray]
    labels: tuple
    mult: tuple
    units: tuple

class BlockPlainMesh(BlockMesh): ...

class BlockPlainVariable(BlockArray):
    grid: BlockPlainMesh
    grid_id: str
    grid_mid: BlockPlainMesh
    data: ndarray
    mult: float
    stagger: int
    units: str

class BlockConstant: ...

class BlockList:
    CPUs_Current_rank: BlockPlainVariable
    CPUs_Original_rank: BlockPlainVariable
    Current_Jx: BlockPlainVariable
    Current_Jy: BlockPlainVariable
    Current_Jz: BlockPlainVariable
    Derived_Average_Particle_Energy: BlockPlainVariable
    Derived_Charge_Density: BlockPlainVariable
    Derived_Number_Density: BlockPlainVariable
    Derived_Number_Density_Protons: BlockPlainVariable
    Derived_Particles_Average_Px: BlockPlainVariable
    Derived_Particles_Average_Px_Protons: BlockPlainVariable
    Derived_Particles_Average_Py: BlockPlainVariable
    Derived_Particles_Average_Py_Protons: BlockPlainVariable
    Derived_Temperature: BlockPlainVariable
    Derived_Temperature_Protons: BlockPlainVariable
    Electric_Field_Ex: BlockPlainVariable
    Electric_Field_Ey: BlockPlainVariable
    Electric_Field_Ez: BlockPlainVariable
    Grid_CPUs_Original_rank: BlockPlainMesh
    Grid_CPUs_Original_rank_mid: BlockPlainMesh
    Grid_Grid: BlockPlainMesh
    Grid_Grid_mid: BlockPlainMesh
    Grid_x_px_Protons: BlockPlainMesh
    Grid_x_px_Protons_mid: BlockPlainMesh
    Header: dict
    Magnetic_Field_Bx: BlockPlainVariable
    Magnetic_Field_By: BlockPlainVariable
    Magnetic_Field_Bz: BlockPlainVariable
    Run_info: dict
    Wall_time: BlockConstant
    dist_fn_x_px_Protons: BlockPlainVariable
