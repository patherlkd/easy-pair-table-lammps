# easy-pair-table-lammps (Currently version-v1)
A relatively simple python tool to create any pair potential to use with the LAMMPS pair_style table option https://docs.lammps.org/pair_table.html, and provides an automatic check against the pair_write output.

This tool does not require you to specify the force! It computes it automatically from the pair function using automatic differentiation.

The automatic check involves automatically running LAMMPS with pair_write and plotting a comparison of your original table (of potential energy and force) against what LAMMPS sees (see below from the Lennard Jones (LJ) make_pairstyle_example_LJ.py example). A text file containing the relative differences is generated (again automatically), helping you to gain a numerical estimate on reliability.

![Plot](test_LJ_plot.png?raw=true "Title")

This tool is currently limited to style = linear (r). But can be easily extended to style = spline (and RSQ or BITMAP) through minor edits to create_lammps_pairstyle_table.py (This will happen in a future version).

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
2. Create a make_pairstyle.py code (copy and edit from make_pairstyle_example_LJ.py).
3. Run the make_pairstyle code `python3 make_pairstyle.py`.
4. Check plot (if plot=True is on) and/or the relative differences data to see if the potential will work in lammps.
5. Run your simulations!

### Look at the make_pairstyle_example_LJ.py

This example code is basic and if you understand how it works you can then do what you want.

## You can get sophisticated!

Note you can specify rather involved potentials, and aslong as you use numpy functions etc. the force can be gotten automatically.
