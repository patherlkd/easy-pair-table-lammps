# easy-pair-table-lammps (Currently version-v2)
A relatively simple python tool to create any pair potential to use with the LAMMPS pair_style table option https://docs.lammps.org/pair_table.html, and provides an automatic check against the pair_write output.

Benefits:

    * One only needs to define a pair function. The force is computed automatically through automatic differentiation.
    * It (the code) automatically makes a plot that compares the original table, with what LAMMPS sees through pair_write.
    * It generates a table containing the relative differences between the original table of energies and forces against pair_write.
    * Users can then simply copy commands from the -- automatically generated -- pair_write input script into their simulation script.


The automatic check involves automatically running LAMMPS with pair_write and plotting a comparison of your original table (of potential energy and force) against what LAMMPS sees (see below for results of the examples). A text file containing the relative differences is generated (again automatically), helping you to gain a numerical estimate on reliability.

This tool has been tested with style = linear and spline. Other options (RSQ and BITMAP etc.) can be introduced through minor edits to create_lammps_pairstyle_table.py.

## How to contribute?

* If you have used this tool and want to add your potential, then send  the python code for such a potential along with the make_<your-potential>.py code and potential and force test plots to luke.davis@ucl.ac.uk. Then I will add it to the potentials_for_lammps.py. Please keep to the style of naming your pair potential function pair_<name>.

## Dependencies:
Linux, python3
### Python packages
* autograd  info: https://autograd.readthedocs.io/en/latest/introduction.html pip: https://pypi.org/project/autograd/

* numpy https://numpy.org/

* scipy https://scipy.org/

* matplotlib (with pdf backend - which it should come with automatically) https://matplotlib.org/

### LAMMPS
You need a lammps executable (you can specify this in the code)

## Main steps

1. Specify functional form of potential in potentials_for_lammps.py (see examples in there for guidance). Just requires the pair-wise distance r to be the first argument. You can add an arbitrary number of other arguments.
2. Create a make_pairstyle.py code (copy and edit from make_pairstyle_example_LJ.py for example).
3. Ensure the path to the lammps executable in the make_pairstyle.py code is correct for your system!
4. Run the make_pairstyle code `python3 make_pairstyle.py`.
5. Check plot (if plot=True is on) and/or the relative differences data to see if the potential will work - as expected - in lammps.
6. Copy the pair_style table and pair_coeff commands from the generated pair_write lammps input script into your actual simulation input scripts.
7. Run your simulations!

### Look at the make_pairstyle_example_LJ.py

This example code is basic and if you understand how it works you can then do what you want. Note that the LJ potential in this example is not shifted or truncated.

## You can get sophisticated!

Note you can specify rather involved potentials, and aslong as you use numpy functions etc. the force can be gotten automatically.

# Example WCA potential

![Plot](Example_pair_WCA.png?raw=true "Title")
