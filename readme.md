# Instructions

## Constants
Constants are stored in `src/hybrid_jp/constants.py`
<!-- 
python -c 'with open("src/hybrid_jp/constants.py", "r") as file:
    lines=file.readlines()
print("\n".join([l.split(" = ")[0] for l in lines]));'
-->
```
VAR_GRID
VAR_GRID_MID
VAR_NUMBERDENSITY
VAR_BX
VAR_BY
VAR_BZ
```

## Command Line Instructions

### Viewing variables contained within an sdf file
```bash
sdf-vars U6T40/0128.cdf
```
outputs:
```bash
CPUs_Current_rank                    [0, 0]
CPUs_Original_rank                   [1, 8]
Current_Jx                           [1600,  160]
Current_Jy                           [1600,  160]
Current_Jz                           [1600,  160]
Derived_Average_Particle_Energy      [1600,  160]
Derived_Charge_Density               [1600,  160]
Derived_Number_Density               [1600,  160]
Derived_Number_Density_Protons       [1600,  160]
Derived_Particles_Average_Px         [1600,  160]
Derived_Particles_Average_Px_Protons [1600,  160]
Derived_Particles_Average_Py         [1600,  160]
Derived_Particles_Average_Py_Protons [1600,  160]
Derived_Temperature                  [1600,  160]
Derived_Temperature_Protons          [1600,  160]
Electric_Field_Ex                    [1600,  160]
Electric_Field_Ey                    [1600,  160]
Electric_Field_Ez                    [1600,  160]
Grid_CPUs_Original_rank              [2, 9]
Grid_CPUs_Original_rank_mid          [1, 8]
Grid_Grid                            [1601,  161]
Grid_Grid_mid                        [1600,  160]
Grid_x_px_Protons                    [1600,  400]
Grid_x_px_Protons_mid                [1599,  399]
Magnetic_Field_Bx                    [1600,  160]
Magnetic_Field_By                    [1600,  160]
Magnetic_Field_Bz                    [1600,  160]
Wall_time                            [1]
dist_fn_x_px_Protons                 [1600,  400]
```
# Changelog