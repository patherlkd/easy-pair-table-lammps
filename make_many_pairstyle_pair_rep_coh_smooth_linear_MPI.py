## Based on code written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk

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
from scipy.signal import argrelmin # to find minima of potentials to extract true minimum 
from functools import partial # to pass functions into functions...
import create_lammps_pairstyle_table as clpt # functions to create and test tabulated pair potential


# add your pairstyles in potentials_for_lammps.py and import it like below :)
from potentials_for_lammps import pair_rep_coh_smooth_linear
from potentials_for_lammps import pair_rep_coh_smooth_linear_scaled
from potentials_for_lammps import force
warnings.simplefilter('ignore')

folderToUse = "/local/ldavis/Phase_Sep_Polymers/pair_tables/" 
os.system("mkdir -p "+folderToUse)

from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

if rank==0:
    clpt.print_version()

# lammps units
units_string = "lj"


# particular arguments for the pairstyle (here the simple pair_LJ)
epr = 20.0 # "kT"
sigma = 1.0
rc = 1.5

# particular options for the table
dist_mode = 2
rmin = 0.5 # minimum distance
rmax = 2.0 # maximum distance
N = 3000 # number of distance values between rmin and rmax to use (note N+1 points get printed!)
res = 5 # resolution (# of decimal places) to round minene, this is should be at least one order of mag lower than O(rmin)*O(N^-1)
Nlmps = N
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

## RANGE OF EPCs ##
epc_delta = 1.0
epcmax = 16.0
epcmin = 0.0
##################

rank_epc_split = (epcmax-epcmin)/float(nprocs)

rank_epcmax = rank_epc_split*float(rank + 1) + epcmin
rank_epcbegin = rank_epc_split*float(rank)+ epcmin
epcs = np.arange(rank_epcbegin,rank_epcmax,epc_delta)

for epc in epcs:
    

    epc = np.around(epc,1)
    # make the intial (pre-scaled) table
    pair_filename = folderToUse+"Table_pair_rep_coh_smooth_linear_temp_"+str(rank)+".txt"
    pair_keyword = "REP_COH_SMOOTH_LINEAR"
    clpt.make_table_for_lammps(pair_filename,pair_keyword,pair_rep_coh_smooth_linear, rmin, rmax, N,rctab, epr, epc, sigma, rc,rctab,mode=dist_mode)
    

    minene = 0.0
    # grab true minimum from the table
    if epc > 0.0:
        pairdata = np.genfromtxt(pair_filename,skip_header=5)

        r = pairdata[:,1]
        ene = pairdata[:,2]

        min_index = argrelmin(ene)[0][0]
        assert(min_index >= 0)

        minene = np.around(ene[min_index],res)
        minr = np.around(r[min_index],res)
        if dist_mode == 2:
            minr = np.around(math.sqrt(r[min_index]),res)
    
    else: # The repulsive case is trivial
        minene = np.around(0.0,res) 
        minr = np.around(sigma*2**(1./6.),res)


    # generate a label for the pair_table with specific parameters
    label = "_sigma"+str(sigma)+"_minr"+str(minr)+"_rc"+str(rc)+"_epr"+str(epr)+"_epc"+str(epc)+"_rmin"+str(rmin)+"_rmax"+str(rmax)+"_Nlmps"+str(Nlmps)

    scale_fact = 1.0
    if epc > 0.0:
        scale_fact = -epc/minene

    #print(minene)
    #print("scale_fact: "+str(scale_fact))
    # make the re-scaled table

    pair_filename = folderToUse+"Table_pair_rep_coh_smooth_linear_scaled"+label+".txt"
    pair_keyword = "REP_COH_SMOOTH_LINEAR_SCALED"
    clpt.make_table_for_lammps(pair_filename,pair_keyword,pair_rep_coh_smooth_linear_scaled, rmin, rmax, N,rctab,scale_fact, epr, epc, sigma, rc,rctab,mode=dist_mode)


    # put the table through lammps and check it
    lmps_pair_write_generic_filename = "./in.pair_write_generic"
    lmps_input_filename = folderToUse+"in.pair_rep_coh_smooth_linear_scaled"+label
    lmps_pair_filename = folderToUse+"Table_lmps_pair_rep_coh_smooth_linear_scaled"+label+".txt"
    lmps_executing_command = "/local/ldavis/lammps/lammps-29Sep2021/src/lmp_serial -i"


    

    # get lammps to pair_write the potential energy and force for comparison
    clpt.pair_write_lammps(lmps_pair_write_generic_filename,lmps_input_filename,lmps_pair_filename,lmps_executing_command,pair_filename, pair_keyword,units_string,rmin,rmax,N+1,Nlmps,rc,style='linear',mode=dist_mode)

    

    # Compare the data in files visually
    clpt.comparison(folderToUse+"pair_rep_coh_smooth_linear_scaled"+label,lmps_pair_filename,pair_filename,rmin,rmax,rdelta,-epc-10,100,-4.0*epc,30,plot=True,markers=False,mode=dist_mode) 
    
    print("(MPI-rank-"+str(rank)+") DONE: "+lmps_pair_filename)

# [ for the above clpt.comparison()] turn plot=True to plot=False to only output file for relative differences between lammps and your original table. In the above test_pair_LJ_rel_differences.txt contains all the numrical difference data and test_pair_LJ.pdf is the plot of the potentials and forces. So the first argument is <some-string> which will make <some-string>_rel_differences.txt and <some-string>.pdf

