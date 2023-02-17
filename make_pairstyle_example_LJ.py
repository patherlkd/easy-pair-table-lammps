

## Written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk

##

##

#!/usr/bin/env python

# coding: utf-8



import warnings
import os
import math
import sys # python system library
import numpy as np # numerical python library
import math # python math library
import scipy # scipy is nearly always useful...
from functools import partial # to pass functions into functions...
import create_lammps_pairstyle_table as clpt # functions to create and test tabulated pair potential


# add your pairstyles in potentials_for_lammps.py and import it like below :)

from potentials_for_lammps import pair_LJ
warnings.simplefilter('ignore')



# lammps units
units_string = "lj"

# particular arguments for the pairstyle
eps = 10.0
sigma = 3.0
rc = 6.0
# particular options for the table

rmin = 0.5 # minimum distance
rmax = 7.0 # maximum distance
N = 500 # number of distance values between rmin and rmax to use
rdelta = (rmax-rmin)/float(N)
rlist = np.arange(rmin,rmax,rdelta)

# define a cut-off due to tabulation
rcdiffp = float('inf')
rctab = float('inf')
for r in rlist:
    if r < rc:
        rcdiff = rc - r
        if rcdiff < rcdiffp:
            rcdiffp = rcdiff
            rctab = r

# make the table

pair_filename = "./Example_pair_LJ.txt"
pair_keyword = "LJ"
clpt.make_table_for_lammps(pair_filename,pair_keyword,pair_LJ, rmin, rmax, N,rctab,eps,sigma,rc)


# put the table through lammps and check it
lmps_input_filename = "./in.Example_lmps_pair_LJ"
lmps_pair_filename = "./Example_lmps_pair_LJ.txt"
lmps_executing_command = "/clusternfs/ldavis/lammps/lammps-29Sep2021/src/lmp_serial -i" # change so it points to your lammps exe




# get lammps to pair_write the potential energy and force for comparison

Nlmps = N
clpt.pair_write_lammps(lmps_input_filename,lmps_pair_filename,lmps_executing_command

                       ,pair_filename, pair_keyword

                       ,units_string,rmin,rmax,N+1,Nlmps,rc,style='linear')



# Compare the data in files visually

clpt.comparison("Example_pair_LJ",lmps_pair_filename,pair_filename,rmin,rmax,rdelta,-20,20,-20,20,plot=True,markers=False) 



# [ for the above clpt.comparison()] turn plot=True to plot=False to only output file for relative differences between lammps and your original table. In the above test_pair_LJ_rel_differences.txt contains all the numrical difference data and test_pair_LJ.pdf is the plot of the potentials and forces. So the first argument is <some-string> which will make <some-string>_rel_differences.txt and <some-string>.pdf

