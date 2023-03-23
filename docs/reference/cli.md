# CLI: hybrid-jp
The base command is `hybrid-jp` - all other commands start with this. 

Based off [Typer](https://typer.tiangolo.com/) and uses [Rich](https://rich.readthedocs.io/) to format and colour the output.

## deck
The base for commands relating to `input.deck` is `hybrid-jp deck`.

### deck eval-consts
Uses [Sympy](https://www.sympy.org/en/index.html) to dynamically solve equations for constants inside the `constant` block in `input.deck`.

!!! info inline end
    Original `input.deck`

        begin:constant
            mp = 1.67e-27
            B0 = 10e-9
            n0 = 10e6

            wci = qe * B0 / mp
            va = ((B0 * B0)/(mu0 * n0 * mp)) ^ 0.5
            di = va / wci
            beta = 1
            T0 = (beta * B0^2) / (2 * mu0 * n0 * kb)

            inflow = 6 * va * mp
            thBn = 40
            ppc = 100
            E0 = inflow * B0 / mp

            amp = 0.5
            sigma = 0.25 * di
        end:constant


```shell
>>> hybrid-jp deck eval-consts path/to/input.deck
begin: constant
      qe  =  1.602e-19
     mu0  =  1.257e-06
      kb  =  1.381e-23
      mp  =  1.670e-27
      B0  =  1.000e-08
      n0  =  1.000e+07
     wci  =  9.594e-01
      va  =  6.903e+04
      di  =  7.195e+04
    beta  =  1.000e+00
      T0  =  2.882e+05
  inflow  =  6.917e-22
    thBn  =  4.000e+01
     ppc  =  1.000e+02
      E0  =  4.142e-03
     amp  =  5.000e-01
   sigma  =  1.799e+04
end: constant
```

## sdf-vars
The base for commands relating to `.sdf` files is `hybrid-jp sdf-vars`.

### sdf-vars read
Reads an sdf file and prints out the variable names along with numpy array sizes.
```shell
>>> hybrid-jp sdf-vars read U6T40/0128.sdf
Reading U6T40/0128.sdf...
Variables:
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