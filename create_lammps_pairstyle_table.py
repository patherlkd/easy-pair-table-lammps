## Written by Dr. Luke Davis: UCL Department of Mathematics 2023 luke.davis@ucl.ac.uk
##
##
#!/usr/bin/env python
# coding: utf-8

import matplotlib
import matplotlib.pyplot as plt # matplotlib shorthand
import matplotlib.ticker as mtick
from matplotlib import cm
matplotlib.use('pdf') # pdf backend

## BEGIN PLOTTING SETTINGS/DETAILS
ff = 3
space         = ff*0.07
nb_lines      = ff*1
fig_width_pt  = 246.
inches_per_pt = 1./72.
golden_mean   = .66
fig_width     = fig_width_pt*inches_per_pt
fig_height    = (fig_width*golden_mean)+space
fig_size      = [ff*fig_width, ff*fig_height]
# Plot parameters                                                                                       
#plt.rcParams['figure.figsize'] = (7,5)                                                                 
#plt.rcParams['figure.dpi'] = 150                                                                       
#plt.rcParams["font.family"] = "Serif"                                                                  
#plt.rcParams["font.size"] = 22                                                                         
#params = {'mathtext.default': 'regular' }                                                              
params = {'legend.fontsize': ff*4,
          'axes.linewidth': ff*6.5e-1,
          #'xaxis.labellocation': 'center',                                                             
          #'yaxis.labellocation': 'center',                                                             
          'axes.labelsize': ff*6,
          'axes.titlesize': ff*6,
          #'text.fontsize': ff*4,                                                                       
          'xtick.labelsize': ff*7,
          'ytick.labelsize': ff*7,
          'text.usetex': False,
          'figure.figsize': fig_size}
plt.rcParams.update(params)
## END PLOTTING SETTINGS/DETAILS

import warnings
import os
import math
import sys # python system library
import autograd.numpy as np # numerical python library
import math # python math library
import scipy # scipy is nearly always useful...
from scipy import interpolate
from functools import partial # to pass functions into functions...

from potentials_for_lammps import force # pair potentials file 
from potentials_for_lammps import pair_debug # simple Lennard-Jones example
from potentials_for_lammps import pair_LJ # simple Lennard-Jones example
warnings.simplefilter('ignore')

MY_VERSION = "v1"

def make_table_for_lammps(filename,pair_keyword,pair,rmin,rmax,N,*args):
    
    rdelta = (rmax-rmin)/float(N) 
    rlist = np.arange(rmin,rmax,rdelta)
    rlist = np.around(rlist,3)
    with open(filename,'w') as file:
        file.write("# Created by [create_lammps_pairstyle_table.py version-"+MY_VERSION+"]\n\n")
        file.write(pair_keyword+"\n")
        file.write("N "+str(N+1)+" R "+str(rlist[0])+" "+str(rlist[-1])+"\n\n")
        n = 1
        for r in rlist:
            file.write(str(n)+' '+str(r)+' '+str(np.around(pair(r,*args),3))+' '+str(np.around(force(partial(pair),r,*args),3))+'\n')
            n+=1


def pair_write_lammps(lmps_input_filename,lmps_pair_filename,lmps_executing_command
                      ,pair_filename, pair_keyword
                      ,units_string,rmin,rmax,N,Nlmps,rc):
    lmpin = open("./in.pair_write_generic",'r')
    lmps_comms = lmpin.read()
    lmpin.close()

    lmps_comms = lmps_comms.replace("UNITS",units_string)
    lmps_comms = lmps_comms.replace("NPOINTS",str(N))
    lmps_comms = lmps_comms.replace("FILE",pair_filename)
    lmps_comms = lmps_comms.replace("KEYWORD",pair_keyword)
    lmps_comms = lmps_comms.replace("RCUT",str(rc))
    lmps_comms = lmps_comms.replace("PWPOINTS",str(Nlmps))
    lmps_comms = lmps_comms.replace("RMIN",str(rmin))
    lmps_comms = lmps_comms.replace("RMAX",str(rmax))
    lmps_comms = lmps_comms.replace("PWFIL",lmps_pair_filename)

    lmpout = open("./"+lmps_input_filename,'w')
    lmpout.write(lmps_comms)
    lmpout.close()
    
    os.system("rm "+lmps_pair_filename+" && "+lmps_executing_command+" "+lmps_input_filename+" > lmps_pair_write_from_create_lammps_pairstyle_table.log") # Let's use os.system() because reasons...


def comparison(file_basename,lmps_pair_filename,pair_filename,xmin,xmax,deltax,y1min,y1max,y2min,y2max,plot=True):
    
    lmpsdata = np.genfromtxt(lmps_pair_filename,skip_header=6)
    OGdata = np.genfromtxt(pair_filename,skip_header=5)
    
    rlmps = lmpsdata[:,1]
    rOG = OGdata[:,1]

    pairlmps = lmpsdata[:,2]
    pairOG = OGdata[:,2]

    forcelmps = lmpsdata[:,3]
    forceOG = OGdata[:,3]
    
    #interpolations
    pairlmps_func = interpolate.interp1d(rlmps,pairlmps)
    forcelmps_func = interpolate.interp1d(rlmps,forcelmps)

    pairOG_func = interpolate.interp1d(rOG,pairOG)
    forceOG_func = interpolate.interp1d(rOG,forceOG)
    
    # create new r data for them to share and we can then compare their values for the same r
    xlist = np.arange(xmin,xmax,deltax)
    
    # save differences to a file
    with open("./"+file_basename+"_rel_differences.txt",'w') as out:
        n = 1
        for x in xlist:
            pairdiff = (pairOG_func(x)-pairlmps_func(x))/pairlmps_func(x)
            if math.isnan(pairdiff):
                pairdiff = 0.0
            forcediff = (forceOG_func(x)-forcelmps_func(x))/forcelmps_func(x)
            if math.isnan(forcediff):
                forcediff = 0.0
            out.write(str(n)+' '+str(x)+' '+str(pairdiff)+' '+str(forcediff)+'\n')
            n += 1
    if plot:
        fig,axs = plt.subplots(1,2,figsize=(14,5))
        
        axs[0].tick_params(which='both',direction='in',left=True,right=False,top=False,bottom=True)
        axs[0].tick_params(axis='both',which='major',length=10,width=2)
        axs[1].tick_params(which='both',direction='in',left=True,right=False,top=False,bottom=True)
        axs[1].tick_params(axis='both',which='major',length=10,width=2)
        
        # plot pair energy first
        axs[0].set_xlim(xmin,xmax)
        axs[0].set_ylim(y1min,y1max)
        
        axs[0].scatter(rlmps,pairlmps,c='blue',marker='s',alpha=1.0,label="pair_write LAMMPS")
        axs[0].plot(rlmps,pairlmps,c='blue',ls='--',alpha=1.0,label="pair_write LAMMPS")
        
        axs[0].scatter(rOG,pairOG,c='black',marker='v',alpha=1.0,label="Original")
        axs[0].plot(rOG,pairOG,c='black',ls='-',alpha=1.0,label="Original")
        axs[0].set(xlabel="r",ylabel="Pair potential energy (r)")
        axs[0].legend(loc='upper right',fontsize=20,markerscale=2.0,frameon=False)
        
        # force
        axs[1].set_xlim(xmin,xmax)
        axs[1].set_ylim(y1min,y1max)
        
        axs[1].scatter(rlmps,forcelmps,c='blue',marker='s',alpha=1.0,label="pair_write LAMMPS")
        axs[1].plot(rlmps,forcelmps,c='blue',ls='--',alpha=1.0,label="pair_write LAMMPS")
        
        axs[1].scatter(rOG,forceOG,c='black',marker='v',alpha=1.0,label="Original")
        axs[1].plot(rOG,forceOG,c='black',ls='-',alpha=1.0,label="Original")
        axs[1].set(xlabel="r",ylabel="Force (r)")
        axs[1].legend(loc='upper right',fontsize=20,markerscale=2.0,frameon=False)
        
        fig.tight_layout(pad=2.0)
        plt.savefig("./"+file_basename+".pdf")
